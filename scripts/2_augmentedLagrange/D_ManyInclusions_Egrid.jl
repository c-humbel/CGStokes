using Printf
using Random
using CairoMakie
using ColorSchemes

include("rectangular_Egrid.jl")


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


function initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; seed=1234, ninc=5)
    rng = MersenneTwister(seed)

    # generate radius and location inclusions
    centers, radii = generate_inclusions(ninc, xc, yc, rng)

    # generate relative viscosity for inclusions
    η_ratios = fill(η_ratio, ninc)
    η_ratios[2:end] .-= rand(rng, ninc-1) .* (η_ratio / 2)
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
    η.c  .= η_mat
    η.v  .= η_mat
    ρg.c .= 0.
    ρg.v .= 0.
    for j = eachindex(yc)
        for i = eachindex(xc)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xc[i] - x)^2 + (yc[j] - y)^2 <= r^2
                    η.c[i, j]  = η_rel * η_mat
                    ρg.c[i, j] = Δρg
                    break
                end
            end
        end
    end

    for j = eachindex(yv)
        for i = eachindex(xv)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xv[i] - x)^2 + (yv[j] - y)^2 <= r^2
                    η.v[i, j]  = η_rel * η_mat
                    ρg.v[i, j] = Δρg
                    break
                end
            end
        end
    end


    return nothing
end

function solve_many_inclusions(;n=127, ninc=8,
                                η_ratio=0.1,
                                niter_in=100, niter_out=100, ncheck=10,
                                γ_factor=1.,
                                ϵ_cg=1e-3, ϵ_ph=1e-6, seed=1234, verbose=false)
    η_avg  = 1. # -> Pa s
    ρg_avg = 1. # -> Pa / m
    Lx     = 1. # -> m

    Ly     = Lx
    R_in   = 0.1 * Lx
    nx, ny = n, n
    dx, dy = Lx / nx, Ly / ny

    xc = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    yc = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)

    # compute viscosities
    A_in  = π * R_in^2
    A_tot = Lx * Ly
    η_out = η_avg * A_tot / (A_tot + (η_ratio - 1)*A_in)
    η_in  = η_out * η_ratio
    # body force
    Δρg   = ρg_avg * A_tot / A_in
    
    # field initialisation
    η    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1)) 
    ρg   = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)
    V    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero

    initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; seed=seed, ninc=ninc)

    # Coefficient of augmented Lagrangian
    γ = γ_factor * max(maximum(η.c), maximum(η.v))

    # preconditioner
    initialise_invM(invM, η, dx, dy, γ)

    # visualisation
    res_out    = []
    res_in     = []
    res_cg     = []
    itercounts = []

    # residual norms for monitoring convergence
    δ     = Inf
    ω     = Inf

    δ_ref = tplNorm(ρg, Inf)
    ω_ref = ρg_avg * Lx / η_avg


    # outer loop, Powell Hestenes
    it_out  = 1
    while it_out <= niter_out && (ω > ϵ_ph || δ > ϵ_ph)
        verbose && println("Iteration ", it_out)
        tplSet!(P₀, P)

        # inner loop, Conjugate Gradient

        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)

        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        # start iteration
        it_in = 0
        while it_in < niter_in
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)

            it_in += 1

            # check convergence
            δ = tplNorm(R, Inf) / δ_ref
            if δ < min(ϵ_cg,  max(ω, ϵ_ph))
                push!(res_cg, δ)
                break
            end
            if it_in % ncheck == 0
                push!(res_cg, δ)
            end
        end
        push!(itercounts, it_in)

        verbose && @printf("CG stopped after %inx iterations\n" , it_in / nx)
        δ > min(ϵ_cg, max(ω , ϵ_ph)) && @printf("CG did not reach prescribed accuracy (%g > %g)\n", δ,  min(ϵ_cg, max(ω , ϵ_ph)))

        push!(res_in, δ)

        compute_divV!(divV, V, dx, dy)
        ω = tplNorm(divV, Inf) / ω_ref
        push!(res_out, ω)

        verbose && println("v-residual = ", δ)
        verbose && println("p-residual = ", ω)

        it_out += 1
    end
    it_out -= 1
    verbose && @printf("Finished after a total of %i outer and %inx CG iterations\n", it_out, sum(itercounts) / nx)
    ω > ϵ_ph && @printf("Iteration did not reach required accuracy (%g > %g)\n", ω,  ϵ_ph)

    return P, V, R, res_out, res_in, res_cg, itercounts, (xc, yc), (xv, yv)
end


n = 128
η_ratio = 1e6
γ_factor = .02
niter_in = 10n*n
niter_out = 500
ncheck = n

outfields = solve_many_inclusions(n=n, ninc=8, η_ratio=η_ratio, niter_in=niter_in, niter_out=niter_out, ncheck=ncheck, ϵ_cg=0.01, verbose=true, seed=2025)

create_output_plot(outfields...; ncheck=ncheck, η_ratio=η_ratio, gamma=γ_factor, savefig=true)