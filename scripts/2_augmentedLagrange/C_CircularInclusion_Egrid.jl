using CairoMakie
using ColorSchemes
using Printf


include("rectangular_Egrid.jl")


function create_output_plot(P, V, R, errs_out, errs_in, errs_cg, itercounts, center_coords, vertex_coords; ncheck, η_ratio, gamma, savefig=false)
    xs, ys = center_coords
    xv, yv = vertex_coords
    dy = ys[2] - ys[1]
    nx = size(P.c, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Residual Norm (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    # compute location of cg iteration errors
    iters_cg  = []
    cg_tot = ncheck
    for it_tot = iters_out
        while cg_tot < it_tot
            push!(iters_cg, cg_tot)
            cg_tot += ncheck
        end
        cg_tot = it_tot
    end
    push!(iters_cg, cg_tot)

    colours = resample(ColorSchemes.viridis,7)[[2, 4, 6]]
    pmax = maximum(abs.(P.c))
    vmax = maximum(abs.(V.yc))


    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=colours[1], marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=colours[2], marker=:diamond, label="Velocity")
    plt = (P=heatmap!(axs.P, xs, ys, P.c, colormap=:PRGn, colorrange=(-pmax, pmax)),
           err=lines!(axs.err, iters_cg ./ nx, log10.(errs_cg), color=colours[3], label="CG"),
           Vy=heatmap!(axs.Vy, xs, yv, V.yc, colormap=:PRGn, colorrange=(-vmax, vmax)),
           Ry=heatmap!(axs.Ry, xs, yv, log10.(abs.(R.yc)), colormap=:viridis))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    Legend(fig[1, 2][2, 1], axs.err, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))

    if savefig
        save("2C_result_$(η_ratio)_$(gamma).png", fig)
    else
        display(fig)
    end
    return nothing
end


function solve_inclusion(;n=127,
                         η_ratio=0.1,
                         niter_in=100, niter_out=100, ncheck=10,
                         γ_factor=1.,
                         ϵ_cg=1e-3, ϵ_ph=1e-6, verbose=false)
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
    η    = (c=[x^2 + y^2 < R_in^2 ? η_in : η_out for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? η_in : η_out for x=xv, y=yv]) 
    ρg   = (c=[x^2 + y^2 < R_in^2 ? Δρg : 0. for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? Δρg : 0. for x=xv, y=yv])
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


    # Coefficient of augmented Lagrangian
    γ = γ_factor * max(η_in, η_out)

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
        δ > min(ϵ_cg, max(ω , ϵ_ph)) && @printf("CG did not reach prescribed accuracy (%g > %g)\n", δ,  min(ϵ_in, max(ω , ϵ_max)))

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
    ω > ϵ_ph && @printf("Iteration did not reach required accuracy (%g > %g)\n", ω,  ϵ_max)

    return P, V, R, res_out, res_in, res_cg, itercounts, (xc, yc), (xv, yv)
end


n = 128
η_ratio = 1e5
γ_factor = .5
niter_in = n*n
niter_out = 500
ncheck = n

outfields = solve_inclusion(n=n, η_ratio=η_ratio, niter_in=niter_in, niter_out=niter_out, ncheck=ncheck, ϵ_cg=0.1, verbose=true)

create_output_plot(outfields...; ncheck=ncheck, η_ratio=η_ratio, gamma=γ_factor, savefig=true)