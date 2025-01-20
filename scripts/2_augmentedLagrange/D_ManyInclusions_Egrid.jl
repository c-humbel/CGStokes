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


function plot_η(η, xs, ys)
    fig = Figure(size=(400, 300))
    ax = Axis(fig[1, 1], aspect=1, xlabel="x", ylabel="y")
    htmp = heatmap!(ax, xs, ys, η, colormap=:viridis)
    Colorbar(fig[1, 2], htmp)
    fig
end


function solve_many_inclusions(;n=127, ninc=8, η_ratio=0.1, γ_factor=1.,
                                max_iter=1e4, ϵ_cg=1e-3, ϵ_ph=1e-6,
                                seed=1234, verbose=false)
    η_avg  = 1. # -> Pa s
    ρg_avg = 1. # -> Pa / m
    Lx     = 1. # -> m

    Ly     = Lx
    nx, ny = n, n
    dx, dy = Lx / nx, Ly / ny

    xc = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    yc = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)
    
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
    itercounts = []

    # residual norms for monitoring convergence
    δ     = Inf
    ω     = Inf

    δ_ref = tplNorm(ρg, Inf)
    ω_ref = ρg_avg * Lx / η_avg


    # outer loop, Powell Hestenes
    it = 0
    while it < max_iter && (ω > ϵ_ph || δ > ϵ_ph)
        tplSet!(P₀, P)

        # inner loop, Conjugate Gradient

        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        δ = tplNorm(R, Inf) / δ_ref
        # start iteration
        while it < max_iter && δ > min(ϵ_cg, max(ω, ϵ_ph))
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)
            δ = tplNorm(R, Inf) / δ_ref

            it += 1
        end

        verbose && @printf("CG stopped after %6.0g iterations: " , isempty(itercounts) ? it : (it - itercounts[end]))

        push!(itercounts, it)

        compute_divV!(divV, V, dx, dy)
        ω = tplNorm(divV, Inf) / ω_ref
        push!(res_out, ω)
        push!(res_in, δ)

        verbose && @printf("v-residual = %12g, p-residual = %12g\n", δ, ω)

        if length(itercounts) >= max_iter it = max_iter end

    end
    if verbose
        @printf("Finished after a total of %i outer and %inx CG iterations\n", length(itercounts), it / nx)
        (ω > ϵ_ph || δ > ϵ_ph) && @printf("Iteration did not reach required accuracy (%g or %g > %g)\n", ω,  δ, ϵ_ph)
    end

    return P, V, R, η, res_out, res_in, itercounts, (xc, yc), (xv, yv)
end


function test_gamma_factor(eta_range=-4:6, gamma_range=LinRange(-2, 1, 10), max_iter=5e5; verbose=false)
    outfile="data_Egrid_gamma_n128_inc8.csv"

    write(outfile, "eta_ratio,eta_max,eta_min,gamma_factor,gamma,iterations\n")

    for η_exp = eta_range
        η_ratio = 10.0^η_exp
        for γ_exp = gamma_range
            γ_factor = round(10.0^γ_exp, sigdigits=1)
            @printf("start with η_ratio %g, γ_factor %g ...%s", η_ratio, γ_factor, verbose ? "\n" : " ")
            results = solve_many_inclusions(n=128, ninc=8, η_ratio=η_ratio, max_iter=max_iter, γ_factor=γ_factor,
                                            ϵ_cg=0.1, ϵ_ph=1e-5, verbose=verbose)
            @printf("done\n\n")
            it      = results[7][end]
            max_eta = tplNorm(results[4], Inf)
            min_eta = tplNorm(results[4], -Inf)
            open(outfile, "a") do io
                @printf(io, "%g,%g,%g,%g,%g,%d\n", η_ratio, max_eta, min_eta, γ_factor, γ_factor * max_eta, it)
            end
        end
    end
end