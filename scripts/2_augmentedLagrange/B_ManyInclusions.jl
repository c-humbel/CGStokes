using Printf
using Random
using CairoMakie
using ColorSchemes

include("rectangular_domain.jl")


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


function initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xs, ys, Lx, Ly; seed=1234, ninc=5)
    rng = MersenneTwister(seed)

    # generate radius and location inclusions
    centers, radii = generate_inclusions(ninc, xs, ys, rng)

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
    η .= η_mat
    ρg .= 0.
    for j = eachindex(ys)
        for i = eachindex(xs)
            for ((xc, yc), r, η_rel) in zip(centers, radii, η_ratios)
                if (xs[i] - xc)^2 + (ys[j] - yc)^2 <= r^2
                    η[i, j] = η_rel * η_mat
                    ρg[i, j] = Δρg
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
    htmp = heatmap!(ax, xs, ys, η', colormap=ColorSchemes.viridis)
    Colorbar(fig[1, 2], htmp)
    fig
end


function plot_results(η, P, V)
    fig = Figure(size=(800, 600))
    axs = (et=Axis(fig[1,1], aspect=1, xlabel="x", ylabel="y", title="Viscosity"),
            P=Axis(fig[1,2], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
           Vx=Axis(fig[2,1], aspect=1, xlabel="x", ylabel="y", title="Horizontal Velocity"),
           Vy=Axis(fig[2,2], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"))
    pmax = 0.5maximum(abs.(P))
    vxmax = maximum(abs.(V.x))
    vymax = maximum(abs.(V.y))
    plt = (et=image!(axs.et, η, colormap=:Greens),
            P=image!(axs.P, P, colormap=:PRGn, colorrange=(-pmax, pmax)),
           Vx=image!(axs.Vx, V.x, colormap=:PRGn, colorrange=(-vxmax, vxmax)),
           Vy=image!(axs.Vy, V.y, colormap=:PRGn, colorrange=(-vymax, vymax)))
    Colorbar(fig[1, 1][1, 2], plt.et)
    Colorbar(fig[1, 2][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vx)
    Colorbar(fig[2, 2][1, 2], plt.Vy)
    fig 
end


function solve_many_inclusions(; n, ninc,
                                η_ratio,
                                niter,
                                γ_factor=1.,
                                ϵ_in=0.1, ϵ_max=1e-5,
                                seed=1234)

    η_avg  = 1. # -> Pa s
    ρg_avg = 1. # -> Pa / m
    Lx     = 1. # -> m

    Ly      = Lx
    nx, ny = n, n
    dx, dy = Lx / nx, Ly / ny

    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # create fields
    η    = zeros(nx, ny)
    ρg   = zeros(nx, ny)
    P    = zeros(nx, ny)
    P₀   = zeros(nx, ny)  # old pressure
    P̄    = zeros(nx, ny)  # memory needed for auto-differentiation
    divV = zeros(nx, ny)
    V    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    V̄    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # memory needed for auto-differentiation
    D    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # preconditioner, cells correspoinding to Dirichlet BC are zero


    # initialisation
    initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xs, ys, Lx, Ly; seed=seed, ninc=ninc)

    # Coefficient of augmented Lagrangian
    γ = γ_factor * maximum(η) # η_avg

    # preconditioner
    initialise_invM(invM, η, dx, dy, γ)

    # residual norms for monitoring convergence
    ω     = Inf
    ω_ref = ρg_avg * Lx / η_avg
    δ_ref = norm(ρg, Inf)

    it = 0

    # outer loop, Powell Hestenes
    while it < niter && (ω > ϵ_max)
        P₀ .= P

        # inner loop, Conjugate Gradient
        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        δ = tplNorm(R, Inf) / δ_ref
        while it < niter && δ > min(ϵ_in, max(ω, ϵ_max))
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)

            # compute residual
            δ = tplNorm(R, Inf) / δ_ref

            it += 1
        end

        # residual based termination
        compute_divV!(divV, V, dx, dy)
        ω = norm(divV, Inf) / ω_ref
    end

    return it, P, V, R, η
end

function test_gamma_factor(outfile="data_gamma_n128_inc8.csv"; eta_range=-4:6, gamma_range=LinRange(-2, 1, 10), niter=5e5)
    write(outfile, "eta_ratio, gamma_factor, iterations")

    for η_exp = eta_range
        η_ratio = 10.0^η_exp
        for γ_exp = gamma_range
            γ_factor = round(10.0^γ_exp, sigdigits=1)
            print("start with η_ratio = $η_ratio, γ_factor = $γ_factor ... ")
            it, _, _, _, _ = solve_many_inclusions(n=128, ninc=8, η_ratio=η_ratio, niter=niter, γ_factor=γ_factor)
            print("finished\n")
            open(outfile, "a") do io
                @printf(io, "%f, %f, %d\n", η_ratio, γ_factor, it)
            end
        end
    end
end