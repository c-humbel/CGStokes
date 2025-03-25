using Printf
using Random
using CairoMakie
using ColorSchemes
using DelimitedFiles

include("../2_augmentedLagrange/rectangular_Egrid.jl")


# copied from PseudoTransientStokes.jl
function generate_inclusions(Lx,Ly,nsub,ri)
    li    = min(3*nsub*ri,Lx-2.5*ri)
    dx    = li/(nsub-1)
    dy    = dx*sqrt(3)/2
    if nsub == 1
        xs   = Float64[0]
        ys   = Float64[0]
        # random offset
        jitx = [0.25*Lx*(2*rand()-1)]
        jity = [0.25*Ly*(2*rand()-1)]
    else
        xs    = Float64[]
        ys    = Float64[]
        ox    = -li/2
        oy    = -li/2*sqrt(3)/2
        for j = 1:nsub
            for i = 1:(nsub-mod(j-1,2))
                push!(xs, ox + (i-1)*dx + dx/2*mod(j-1,2))
                push!(ys, oy + (j-1)*dy)
            end
        end
        # random offset
        jitx = 0.5*(dx-2*ri) .* (2 .* rand(length(xs)) .- 1.0)
        jity = 0.5*(dy-2*ri) .* (2 .* rand(length(xs)) .- 1.0)
        #exponenial falloff
        idx = xs.*jitx .> 0.0
        idy = ys.*jity .> 0.0
        jitx[idx] .*= exp.(-(xs[idx]/2).^2)
        jity[idy] .*= exp.(-(ys[idy]/2).^2)
    end
    # jitter
    xs .+= jitx
    ys .+= jity
    return xs,ys
end

function initialise_η_ρg!(η, ρg, xi, yi, r, η0, ηi, ρg0, ρgi, xc, yc, xv, yv)
    tplFill!(η, η0)
    fill!(ρg.c, ρg0)
    fill!(ρg.v, ρg0)

    for j = eachindex(yc)
        for i = eachindex(xc)
            for (x, y) ∈ zip(xi, yi)
                if (xc[i] - x)^2 + (yc[j] - y)^2 <= r^2
                    η.c[i, j]  = ηi
                    ρg.c[i, j]  = ρgi
                    break
                end
            end
        end
    end

    for j = eachindex(yv)
        for i = eachindex(xv)
            for (x, y) ∈ zip(xi, yi)
                if (xv[i] - x)^2 + (yv[j] - y)^2 <= r^2
                    η.v[i, j]  = ηi
                    ρg.v[i, j]  = ρgi
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


function solve_many_inclusions(;n=127, nsub=8, vrpow=1, γ_factor=1., nout=50,
                                max_iter=1e4, ϵ_cg=1e-3, ϵ_ph=1e-6,
                                verbose=false)
    Random.seed!(1855 + nsub)
    nx, ny = n, n
    Lx, Ly = 10.0, 10.0          # domain extends
    dx, dy = Lx / nx, Ly / ny
    η0     = 1.0                 # matrix viscosity
    ηi     = 10.0^(-vrpow)       # inclusion viscosity
    ρg0    = 0. 
    ρgi    = 1.    
    ri     = sqrt(Lx*Ly*0.005/π)
    εbg    = 1.

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

    xi, yi = generate_inclusions(Lx, Ly, nsub, ri)
    initialise_η_ρg!(η, ρg, xi, yi, ri, η0, ηi, ρg0, ρgi, xc, yc, xv, yv)

    # V.xc .= -εbg.*[((ix-1)*dx -0.5*Lx) for ix=1:size(V.xc,1), iy=1:size(V.xc,2)]
    # V.yc .=  εbg.*[((iy-1)*dy -0.5*Ly) for ix=1:size(V.yc,1), iy=1:size(V.yc,2)]
    # V.xv .= -εbg.*[((ix-1.5)*dx -0.5*Lx) for ix=1:size(V.xv,1), iy=1:size(V.xv,2)]
    # V.yv .=  εbg.*[((iy-1.5)*dy -0.5*Ly) for ix=1:size(V.yv,1), iy=1:size(V.yv,2)]

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
    ω_ref = tplNorm(ρg, 1) * Lx / tplNorm(η, 1)


    # outer loop, Powell Hestenes
    verbose && @printf("Starting iteration\n")
    wtime = 0
    it = 0
    while it < max_iter && (ω > ϵ_ph || δ > ϵ_ph)
        tplSet!(P₀, P)

        # inner loop, Conjugate Gradient

        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        δ = Inf
        # start iteration
        while it < max_iter && δ > min(ϵ_cg, max(ω, ϵ_ph))
            if it == 11 wtime = -Base.time() end
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)
            if it % nout == 0
                Pmax, Pmin = max(maximum(P.c), maximum(P.v)), min(minimum(P.c), minimum(P.v))
                normRx = sqrt(norm(R.xc)^2 + norm(R.xv)^2) / (Pmax - Pmin) * Lx / sqrt(length(R.xc) + length(R.xv))
                normRy = sqrt(norm(R.yc)^2 + norm(R.yv)^2) / (Pmax - Pmin) * Ly / sqrt(length(R.yc) + length(R.yv))
                δ = max(normRx, normRy)
            end

            it += 1
        end

        verbose && @printf("CG stopped after %i iterations: " , isempty(itercounts) ? it : (it - itercounts[end]))

        push!(itercounts, it)


        Vmax, Vmin = max(maximum(V.yc), maximum(V.yv)), min(minimum(V.yc), minimum(V.yv))
        compute_divV!(divV, V, dx, dy)
        ω = tplNorm(divV) / (Vmax - Vmin) * Lx / sqrt(length(divV.c) + length(divV.v))
        push!(res_out, ω)
        push!(res_in, δ)

        verbose && @printf("v-residual = %12g, p-residual = %12g\n", δ, ω)

        if length(itercounts) >= max_iter it = max_iter end

    end
    wtime += Base.time()
    if verbose
        @printf("Finished after a total of %i outer and %i CG iterations\n", length(itercounts), it)
        @printf("Total time: %g\n", wtime)
        (ω > ϵ_ph || δ > ϵ_ph) && @printf("Iteration did not reach required accuracy (%g or %g > %g)\n", ω,  δ, ϵ_ph)
    end

    return P, V, η, res_out, res_in, itercounts, xc, yc, xv, yv
end


P, V, η, res_out, res_in, itercounts, xc, yc, xv, yv = solve_many_inclusions(n=255, nsub=2, vrpow=-4, γ_factor=.5,
                                                        max_iter=500*255, ϵ_cg=0.1, ϵ_ph=1e-8,
                                                        verbose=true);


data_pt = readdlm("data_pt.csv", ',', Float64)

with_theme(theme_latexfonts()) do 
    violet, green = resample(ColorSchemes.viridis, 5)[[1, 3]]
    nx  = length(xc)
    fig = Figure(fontsize=16,size=(600,600))
    axs = (pt=Axis(fig[1,1], title="Pseudo-Transient Method", ylabel="residual norm", yscale=log10),
           cg=Axis(fig[2,1], title="Powell-Hestenes with Conjugate Gradients",ylabel="residual norm",  xlabel="iterations / nx", yscale=log10))
    linkaxes!(axs.pt, axs.cg)
    
    lines!(axs.pt, data_pt[:, 1] ./ nx, data_pt[:, 2], color=violet, label="velocity")
    scatterlines!(axs.cg, itercounts ./ nx, res_in, color=violet, marker=:circle, label="velocity", linestyle=:dash)
    plt = (pt=lines!(axs.pt, data_pt[:, 1] ./ nx, data_pt[:, 3], color=green, label="pressure"),
           cg=scatterlines!(axs.cg, itercounts ./ nx, res_out, color=green, marker=:diamond, label="pressure", linestyle=:dash))
    
    Legend(fig[3,1], axs.cg, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))
    save("comparison.pdf", fig)
end

with_theme(theme_latexfonts()) do
    violet, green = resample(ColorSchemes.viridis, 5)[[1, 3]]
    nx  = length(xc)
    fig = Figure(fontsize=16,size=(800,800))
    axs = (eta=Axis(fig[1,1][1,1], title="Viscosity", xlabel="x", ylabel="y", aspect=1),
            Pc=Axis(fig[2,1][1,1], title="Pressure", xlabel="x", ylabel="y", aspect=1),
            Vy=Axis(fig[1,2][1,1], title="Horizontal Velocity", xlabel="x", ylabel="y", aspect=1),
            err=Axis(fig[2,2][1,1], xlabel="Iterations / nx", title="Residual Norm (log)"),
        )

    scatterlines!(axs.err, itercounts ./ nx, log10.(res_out), color=violet, marker=:circle, label="Pressure")

    plt = (eta=heatmap!(axs.eta, xc, yc, η.c, colormap=:viridis),
            Pc=heatmap!(axs.Pc, xc, yc, P.c, colormap=ColorSchemes.viridis),
            Vy=heatmap!(axs.Vy, xc, yv, V.yc, colormap=ColorSchemes.viridis),
            err=scatterlines!(axs.err, itercounts ./ nx, log10.(res_in), color=green, marker=:diamond, label="Velocity"))

    cbar= (eta=Colorbar(fig[1,1][1, 2], plt.eta),
            Pc=Colorbar(fig[2,1][1, 2], plt.Pc),
            Vy=Colorbar(fig[1,2][1, 2], plt.Vy),
            err=Legend(fig[2,2][2, 1], axs.err, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0)))

    save("result_cg.pdf", fig)
end