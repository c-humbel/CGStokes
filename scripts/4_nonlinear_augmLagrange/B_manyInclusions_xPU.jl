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
function initialise_B_ρ!(B, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; seed=1234, ninc=5)
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
    η_loc  = (c=Array(B.c), v=Array(B.v))
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
    
    copyto!(B.c, η_loc.c)
    copyto!(B.v, η_loc.v)
    copyto!(ρg.c, ρg_loc.c)
    copyto!(ρg.v, ρg_loc.v)

    return nothing
end


function nonlinear_inclusion(;n=127, ninc=5, η_ratio=0.1, niter=10000, γ_factor=1.,
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
    divV = deepcopy(P)  # velocity divergence
    ρg   = deepcopy(P)  # body force
    B    = deepcopy(P)  # prefactor of constituitive relation
    ϵ̇_E  = deepcopy(P)  # invariant of strain rate
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx, ny)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))  # deviatoric stress tensor
    τ̄    = deepcopy(τ)
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

    
    initialise_B_ρ!(B, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; ninc=ninc)

    γ = γ_factor * tplNorm(B, Inf)


    # residual norms for monitoring convergence
    δ = Inf # CG
    χ = Inf # Newton
    ω = Inf # Pressure

    
    δ_ref = tplNorm(ρg, Inf) # is this correct ?
    ω_ref = ρg_avg * Lx / η_avg

    # visualisation
    itercounts = []
    res_newton = []

    fig = Figure(size=(800,900))
    axs = (Bc=Axis(fig[1,1][1,1], aspect=1, title="viscosity prefactor"),
           Pc=Axis(fig[1,2][1,1], aspect=1, title="pressure"),
           Vx=Axis(fig[2,1][1,1], aspect=1, title="horizontal velocity"),
           Vy=Axis(fig[2,2][1,1], aspect=1, title="vertical velocity"),
           Sr=Axis(fig[3,1][1,1], aspect=1, title="strain rate"),
           Er=Axis(fig[3,2][1,1], title="Convergence of Newton Method", xlabel="iterations / nx", ylabel="residual norm"))

    plt = (Bc=heatmap!(axs.Bc, B.c, colormap=ColorSchemes.viridis),
           Pc=heatmap!(axs.Pc, P.c, colormap=ColorSchemes.viridis),
           Vx=heatmap!(axs.Vx, V.xc, colormap=ColorSchemes.viridis),
           Vy=heatmap!(axs.Vy, V.yc, colormap=ColorSchemes.viridis),
           Sr=heatmap!(axs.Sr, ϵ̇_E.c, colormap=ColorSchemes.viridis))
    cbar= (B=Colorbar(fig[1, 1][1, 2], plt.Bc),
           P=Colorbar(fig[1, 2][1, 2], plt.Pc),
           Vx=Colorbar(fig[2, 1][1, 2], plt.Vx),
           Vy=Colorbar(fig[2, 2][1, 2], plt.Vy),
           Sr=Colorbar(fig[3, 1][1, 2], plt.Sr))

    display(fig)

    # create Kernels
    init_invM! = initialise_invM(backend, workgroup, (nx+2, ny+2))
    up_D!      = update_D!(backend, workgroup, (nx+2, ny+2))
    up_V!      = update_V!(backend, workgroup, (nx+2, ny+2))
    comp_divV! = compute_divV!(backend, workgroup, (nx+1, ny+1))
    comp_P_τ   = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
    comp_R     = compute_R!(backend, workgroup, (nx+2, ny+2))
    comp_ϵ̇_E   = compute_strain_rate!(backend, workgroup, (nx+1, ny+1))

    # create function for jacobian-vector product
    
    # compute D_v r * V̄ as D_p r * D_v p * V̄ + D_τ r * D_v τ * V̄
    function jvp_R(R, Q, P, P̄, τ, τ̄, V, V̄, P₀, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        autodiff(Forward, comp_P_τ, DuplicatedNoNeed(P, P̄), DuplicatedNoNeed(τ, τ̄),
                 Const(P₀), Duplicated(V, V̄), Const(B),
                 Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R, DuplicatedNoNeed(R, Q),
                 Duplicated(P, P̄), Duplicated(τ, τ̄), Const(ρg), Const(iΔx), Const(iΔy))
    
    end


    # Powell Hestenes
    it = 0
    while it < niter && ω > ϵ_ph
        verbose && println("Iteration ", it_P)
        # p_0 = p
        tplSet!(P₀, P)

        # r = f - div τ + grad p
        comp_P_τ(P, τ, P₀, V, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        comp_R(R, P, τ, ρg, iΔx, iΔy)

        χ = tplNorm(R, Inf) / δ_ref

        # Newton iteration
        while it < niter && χ > ϵ_newton
            # initialise preconditioner
            # ϵ̇_E = 0.5 * ϵ̇_ij * ϵ̇_ij
            comp_ϵ̇_E(ϵ̇_E, V, iΔx, iΔy, ϵ̇_bg)
            init_invM!(invM, ϵ̇_E, B, q, iΔx, iΔy, γ)

            # iteration zero
            # compute residual for CG,
            # k = r - D_v r * dv
            tplSet!(V̄, dV)
            jvp_R(K, Q, P, P̄, τ, τ̄, V, V̄, P₀, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            tplSet!(K, R)
            tplSub!(K, Q)

            # D = inv(M) * k
            tplSet!(D, K, invM)
            μ = tplDot(K, D)
            δ = tplNorm(K, Inf) / δ_ref
            # start iteration
            while it <= niter && δ > ϵ_cg
                # compute α
                # α = k^T * inv(M) * k / (d^T * D_v r * d)
                tplSet!(V̄, D)
                jvp_R(K, Q, P, P̄, τ, τ̄, V, V̄, P₀, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)
                α = μ / tplDot(D, Q)

                # dv += α d
                up_V!(dV, D, α)

                # recompute residual
                # k = r - D_v r * dv
                tplSet!(V̄, dV)
                jvp_R(K, Q, P, P̄, τ, τ̄, V, V̄, P₀, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)
                tplSet!(K, R)
                tplSub!(K, Q)

                # μ = k^T inv(M) k
                μ_new = tplDot(K, K, invM)
                β = μ_new / μ
                μ = μ_new
                # d = β d + inv(M) * k 
                up_D!(D, K, invM, β)

                # compute residual norm
                δ = tplNorm(K, Inf) / δ_ref # correct scaling?
                it += 1

                if verbose && it % 100 == 0 println("CG residual = ", δ) end
            end
            # damped to newton iteration
            # find λ st. r(v + λ dv) < r(v)
            tplSet!(V̄, V)
            # tplScale!(dV, λ)
            tplAdd!(V̄, dV)

            comp_P_τ(P, τ, P₀, V̄, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            comp_R(R, P, τ, ρg, iΔx, iΔy)

             χ_new = tplNorm(R, Inf) / δ_ref
             λ = 1.
             while χ_new >= χ && λ > 1e-4
                tplSet!(V̄, V)
                tplScale!(dV, inv(MathConstants.golden))
                tplAdd!(V̄, dV)
                comp_P_τ(P, τ, P₀, V̄, B, q, ϵ̇_bg, iΔx, iΔy, γ)
                comp_R(R, P, τ, ρg, iΔx, iΔy)
                λ /= MathConstants.golden
                χ_new = tplNorm(R, Inf) / δ_ref
             end
             tplSet!(V, V̄)
             # λ *= MathConstants.golden
             χ = χ_new
 
             push!(itercounts, it)
             push!(res_newton, χ)
 
             # update plot -> works only for cpu backend
             comp_ϵ̇_E(ϵ̇_E, V, iΔx, iΔy, ϵ̇_bg)
             plt.Pc[3][] .= P.c
             plt.Vx[3][] .= V.xc
             plt.Vy[3][] .= V.yc
             plt.Sr[3][] .= log10.(ϵ̇_E.c)
             plt.Pc.colorrange[] = (min(-1e-10,minimum(P.c )), max(1e-10,maximum(P.c )))
             plt.Vx.colorrange[] = (min(-1e-10,minimum(V.xc)), max(1e-10,maximum(V.xc)))
             plt.Vy.colorrange[] = (min(-1e-10,minimum(V.yc)), max(1e-10,maximum(V.yc)))
             plt.Sr.colorrange[] = (min(-1,log10(minimum(ϵ̇_E.c))), max(1,log10(maximum(ϵ̇_E.c))))
 
             scatterlines!(axs.Er, itercounts ./ nx, log10.(res_newton), color=:purple)
 
             display(fig)
             
             println("Newton residual = ", χ, "; λ = ", λ, "; total iteration count: ", it)
         end    
         comp_divV!(divV, V, iΔx, iΔy)
         ω = tplNorm(divV, Inf) / ω_ref # correct scaling?
         println("Pressure residual = ", ω, ", Newton residual = ", χ, ", CG residual = ", δ)
     end
 
     return it, P, V, R
end


outfields = nonlinear_inclusion(n=64, ninc=4, η_ratio=5.,γ_factor=100., niter=6000, ϵ_ph=1e-3, ϵ_cg=1e-3, ϵ_newton=1e-3);

