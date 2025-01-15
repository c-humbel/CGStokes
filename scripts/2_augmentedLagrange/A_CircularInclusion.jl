using Printf
using CairoMakie
using ColorSchemes

include("rectangular_domain.jl")

function create_output_plot(P, V, R, errs_in, errs_out, conv_cg, itercounts, xs, ys; ncheck, η_ratio, gamma, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Scaled Residuals (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    # compute location of cg iteration errors
    iters_cg  = []
    cg_tot = 0
    for it_tot = iters_out
        while cg_tot < it_tot
            push!(iters_cg, cg_tot)
            cg_tot += ncheck
        end
        cg_tot = it_tot
        push!(iters_cg, cg_tot)
    end

    colours = resample(ColorSchemes.viridis,7)[[2, 4, 6]]
    # color limits
    pmax = maximum(abs.(P))
    vmax = maximum(abs.(V.y))
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:PRGn, colorrange=(-pmax, pmax)),
           err=lines!(axs.err, iters_cg ./ nx, log10.(conv_cg), color=colours[3], label="CG conv."),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:PRGn, colorrange=(-vmax, vmax)),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:viridis))

    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=colours[1], marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=colours[2], marker=:diamond, label="Velocity")

    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    Legend(fig[1, 2][2, 1], axs.err, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))

    if savefig
        save("2_output_$(η_ratio)_$(gamma).png", fig)
    else
        display(fig)
    end
    return nothing
end

function solve_inclusion(; n,
                           η_ratio,
                           niter_in, niter_out, ncheck,
                           γ_factor=1.,
                           ϵ_in=0.1,ϵ_max=1e-5,
                           verbose=false)
    η_avg  = 1. # -> Pa s
    ρg_avg = 1. # -> Pa / m
    Lx     = 1. # -> m

    nx = ny = n
    R_in    = 0.1 * Lx
    Ly      = Lx

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)


    # compute viscosities
    A_in  = π * R_in^2
    A_tot = Lx * Ly
    η_out = η_avg * A_tot / (A_tot + (η_ratio - 1)*A_in)
    η_in  = η_out * η_ratio
    # body force
    Δρg   = ρg_avg * A_tot / A_in

    # field initialisation
    η    = [x^2 + y^2 < R_in^2 ? η_in : η_out for x=xs, y=ys]  
    ρg   = [x^2 + y^2 < R_in^2 ? Δρg : 0. for x=xs, y=ys]
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


    # Coefficient of augmented Lagrangian
    γ = γ_factor * max(η_in, η_out) # η_avg

    # preconditioner
    initialise_invM(invM, η, dx, dy, γ)

    # visualisation
    res_out    = []
    res_in     = []
    conv_in    = []
    itercounts = []

    # residual norms for monitoring convergence
    ω     = Inf
    ω_ref = ρg_avg * Lx / η_avg
    δ_ref = norm(ρg, Inf)

    # outer loop, Powell Hestenes
    it_out  = 1
    while it_out <= niter_out && (ω > ϵ_max)
        verbose && @printf("Iteration %i\n", it_out)
        P₀ .= P

        # inner loop, Conjugate Gradient

        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)

        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        # δ = sqrt(μ) / δ_ref
        δ = tplNorm(R, Inf) / δ_ref
        # δ = tplNorm(D, Inf) / tplNorm(R, Inf)
        push!(conv_in, δ)
        # start iteration
        it_in = 1
        while it_in <= niter_in
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)

            # check convergence
            # δ = α * tplNorm(D, Inf) / tplNorm(R, Inf)
            # δ = sqrt(μ) / δ_ref
            δ = tplNorm(R, Inf) / δ_ref
            if δ < min(ϵ_in, max(ω, ϵ_max))
                push!(conv_in, δ)
                it_in += 1
            break
            end
            if it_in % ncheck == 0
                push!(conv_in, δ)
            end
            it_in += 1
        end
        it_in -= 1
        push!(itercounts, it_in)
        verbose && @printf("CG stopped after %inx iterations\n" , it_in / nx)
        δ > min(ϵ_in, max(ω , ϵ_max)) && @printf("CG did not reach prescribed accuracy (%g > %g)\n", δ,  min(ϵ_in, max(ω , ϵ_max)))

        # residual based termination
        compute_divV!(divV, V, dx, dy)
        ω = norm(divV, Inf) / ω_ref

        # record residuals
        push!(res_in, δ)
        push!(res_out, ω)

        verbose && @printf("ω = %g, δ = %g\n", ω, δ)
        it_out += 1
    end
    it_out -= 1
    verbose && @printf("Finished after a total of %i outer and %inx CG iterations\n", it_out, sum(itercounts) / nx)
    ω > ϵ_max && @printf("Iteration did not reach required accuracy (%g > %g)\n", ω,  ϵ_max)

    return P, V, R, res_in, res_out, conv_in, itercounts, xs, ys
end