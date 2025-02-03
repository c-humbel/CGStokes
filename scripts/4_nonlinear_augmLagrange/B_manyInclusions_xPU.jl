using CairoMakie
using ColorSchemes
using Enzyme
using KernelAbstractions
using Random

include("../../src/tuple_manip.jl")
include("kernels_free_slip.jl")


# copied from 2_augmentedLagrange/D_ManyInclusions_Egrid.jl
function generate_inclusions(ninc, xs, ys, rng)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    nx = length(xs)
    ny = length(ys)
    Lx = nx * dx
    Ly = ny * dy

    r_min = 2   * max(dx, dy)
    r_max = 0.1 * min(Lx, Ly)

    # generate random radii
    rs = r_min .+ (r_max - r_min) .* rand(rng, ninc)

    # generate random positions for non-overlapping circles
    xcs = zeros(ninc)
    ycs = zeros(ninc)
    i = 1
    while i <= ninc
        # generate guess
        xcs[i] = rand(rng, xs[end÷5:4end÷5])
        ycs[i] = rand(rng, ys[end÷5:4end÷5])
        # check that cicles are not overlapping with existing ones
        j = 1
        while j < i
            if (xcs[i] - xcs[j])^2 + (ycs[i] - ycs[j])^2 < (rs[i] + rs[j] + 2r_min)^2
                break
            end
            j += 1
        end     
        if j == i
            i += 1
        end 
    end
    return zip(xcs, ycs), rs
end

# copied from 2_augmentedLagrange/D_ManyInclusions_Egrid.jl
function initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; seed=1234, ninc=5)
    rng = MersenneTwister(seed)

    # generate radius and location inclusions
    centers, radii = generate_inclusions(ninc, xc, yc, rng)

    # generate relative viscosity for inclusions
    η_ratios = fill(η_ratio, ninc)
    offsets = rand(rng, ninc-1) .* (η_ratio / 2)
    if η_ratio > 1
        η_ratios[2:end] .-= offsets
    elseif η_ratio < 1
        η_ratios[2:end] .+= offsets
    end
    shuffle!(rng, η_ratios)

    # area of inclusions
    As = [π*r^2 for r in radii]
    A_inc = sum(As)
    A_tot = Lx * Ly

    # matrix viscosity
    η_mat = η_avg * A_tot / (sum(As .* η_ratios) + A_tot - A_inc)

    # body force
    Δρg   = ρg_avg * A_tot / A_inc

    # set viscosity and body force values
    η_loc  = (c=Array(η.c), v=Array(η.v))
    ρg_loc = (c=Array(ρg.c), v=Array(ρg.v))
    η_loc.c  .= η_mat
    η_loc.v  .= η_mat
    ρg_loc.c .= 0.
    ρg_loc.v .= 0.
    for j = eachindex(yc)
        for i = eachindex(xc)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xc[i] - x)^2 + (yc[j] - y)^2 <= r^2
                    η_loc.c[i, j]  = η_rel * η_mat
                    ρg_loc.c[i, j] = Δρg
                    break
                end
            end
        end
    end

    for j = eachindex(yv)
        for i = eachindex(xv)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xv[i] - x)^2 + (yv[j] - y)^2 <= r^2
                    η_loc.v[i, j]  = η_rel * η_mat
                    ρg_loc.v[i, j] = Δρg
                    break
                end
            end
        end
    end
    
    copyto!(η.c, η_loc.c)
    copyto!(η.v, η_loc.v)
    copyto!(ρg.c, ρg_loc.c)
    copyto!(ρg.v, ρg_loc.v)

    return nothing
end


function nonlinear_inclusion(;n=127, η_ratio=0.1, niter=10000, γ=1.,
                            ϵ_cg=1e-3, ϵ_ph=1e-6, ϵ_newton=1e-3,
                            backend=CPU(), workgroup=64, type=Float64, verbose=false)
    L_ref =  1. # reference length 
    ρg_avg = 1. # average density

    # physical parameters would be: n == 3, A = (24 * 1e-25) in Glen's law, see Cuffrey and Paterson (2006), table 3.3
    # and q = 1. + 1/n, η_avg = (24 * 1e-25) ^ (-1/n), see Schoof (2006)
    η_avg = 1. 
    q = 1. + 1/3  

    Lx = Ly = L_ref
    nx = ny = n
    ϵ̇_bg = eps(type)

    Δx,  Δy  = Lx / nx, Ly / ny
    iΔx, iΔy = inv(Δx), inv(Δy)
    xc = LinRange(-0.5Lx + 0.5Δx, 0.5Lx - 0.5Δx, nx)
    yc = LinRange(-0.5Ly + 0.5Δy, 0.5Ly - 0.5Δy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)


    # field initialisation
    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny),
            v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)
    ρg   = deepcopy(P)
    B    = deepcopy(P)
    η    = deepcopy(P)  # viscosity
    η̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    dV   = deepcopy(V)  # velocity updates in Newton iteration
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # nonlinear Residual
    K    = deepcopy(V)  # Residuals in CG
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by some vector (used for autodiff)
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly)

    # residual norms for monitoring convergence
    δ = Inf # CG
    χ = Inf # Newton
    ω = Inf # Pressure

    δ_ref = tplNorm(ρg, Inf) # is this correct ?
    ω_ref = ρg_avg * Lx / η_avg

    # visualisation
    fig = Figure(size=(600,400))
    axs = (Eta=Axis(fig[1,1][1,1], aspect=1), P=Axis(fig[1,2][1,1], aspect=1),
           Vx=Axis(fig[2,1][1,1], aspect=1), Vy=Axis(fig[2,2][1,1], aspect=1))
    plt = (Eta=heatmap!(axs.Eta, Array(η.c), colormap=ColorSchemes.viridis),
           P=heatmap!(axs.P, Array(P.c), colormap=ColorSchemes.viridis),
           Vx=heatmap!(axs.Vx, Array(V.xc), colormap=ColorSchemes.viridis),
           Vy=heatmap!(axs.Vy, Array(V.yc), colormap=ColorSchemes.viridis))
    Colorbar(fig[1, 1][1, 2], plt.Eta)
    Colorbar(fig[1, 2][1, 2], plt.P),
    Colorbar(fig[2, 1][1, 2], plt.Vx),
    Colorbar(fig[2, 2][1, 2], plt.Vy)

    display(fig)

    # create Kernels
    init_invM! = initialise_invM(backend, workgroup, (nx+2, ny+2))
    up_D!      = update_D!(backend, workgroup, (nx+2, ny+2))
    up_V!      = update_V!(backend, workgroup, (nx+2, ny+2))
    comp_divV! = compute_divV!(backend, workgroup, (nx+1, ny+1))

    # create function 


    # Powell Hestenes
    it = 0
    while it < niter && ω > ϵ_ph
        verbose && println("Iteration ", it_P)
        tplSet!(P₀, P)

        comp_R!(R, P,  η, P₀, V, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)

        χ = tplNorm(R, Inf) / δ_ref

        # Newton iteration
        while it < niter && χ > ϵ_newton
            # initialise preconditioner
            init_invM!(invM, η, iΔx, iΔy, γ)

            # iteration zero
            # compute residual for CG, K = R - DR * dV
            tplSet!(V̄, dV)
            autodiff(Forward, comp_R!, DuplicatedNoNeed(R, Q),
                     Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                     Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))
            tplSet!(K, R)
            tplAdd!(K, Q)

            tplSet!(D, K, invM)
            μ = tplDot(K, D)
            δ = tplNorm(K, Inf) / δ_ref
            # start iteration
            while it <= niter && δ > ϵ_cg
                # compute α
                tplSet!(V̄, D)
                autodiff(Forward, comp_R!, DuplicatedNoNeed(K, Q),
                     Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                     Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

                α = - μ / tplDot(D, Q)

                up_V!(dV, D, α)

                # recompute residual
                tplSet!(V̄, dV)
                autodiff(Forward, comp_R!, DuplicatedNoNeed(R, Q),
                         Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                         Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))
                tplSet!(K, R)
                tplAdd!(K, Q)


                μ_new = tplDot(K, K, invM)
                β = μ_new / μ
                μ = μ_new
                up_D!(D, K, invM, β)

                # compute residual norm
                δ = tplNorm(K, Inf) / δ_ref # correct scaling?
                it += 1

                if it % 10 == 0 println("CG residual = ", δ) end
            end
            tplAdd!(V, dV)

            # update plot
            plt.Eta[3][] .= log10.(η.c)
            plt.P[3][]   .= P.c
            plt.Vx[3][]  .= V.xc
            plt.Vy[3][]  .= V.yc
            plt.Eta.colorrange[]= (min(-1,log10(minimum(η.c))), max(1,log10(maximum(η.c))))
            plt.P.colorrange[]  = (min(-1e-10,minimum(P.c)), max(1e-10, maximum(P.c)))
            plt.Vx.colorrange[] = (min(-1e-10,minimum(V.xc)), max(1e-10,maximum(V.xc)))
            plt.Vy.colorrange[] = (min(-1e-10,minimum(V.yc)), max(1e-10,maximum(V.yc)))


            display(fig)

            comp_R!(R, P,  η, P₀, V, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ,)
            χ = tplNorm(R, Inf) / δ_ref # correct scaling?
            println("Newton residual = ", χ, "; total iteration count: ", it)
        end    
        comp_divV!(divV, V, iΔx, iΔy,)
        ω = tplNorm(divV, Inf) / ω_ref # correct scaling?
        println("Pressure residual = ", ω, ", Newton residual = ", χ, ", CG residual = ", δ)
    end

    return it, P, V, R, η
end


outfields = nonlinear_inclusion(n=64, niter=5000, γ=10., ϵ_ph=1e-3, ϵ_cg=1e-3, ϵ_newton=0.5);

