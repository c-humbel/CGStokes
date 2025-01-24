using CairoMakie
using ColorSchemes
using Printf

function load_data(filename, maxiter=5e5)
    eta_ratios = []
    gamma_factors = []
    iterations = []
    max_etas  = []
    min_etas  = []
    gammas = []
    

    for (i, line) ∈ enumerate(eachline(filename))
        if i == 1 continue end
        η_ratio, max_η, min_η, γ_factor, γ, it = split(line, ",")
        push!(eta_ratios, parse(Float64, η_ratio))
        push!(gamma_factors, parse(Float64, γ_factor))
        push!(iterations, parse(Int, it))
        push!(max_etas, parse(Float64, max_η))
        push!(min_etas, parse(Float64, min_η)) 
        push!(gammas, parse(Float64, γ))
    end

    unique!(eta_ratios)
    unique!(gamma_factors)
    unique!(max_etas)
    unique!(min_etas)


    @assert length(eta_ratios) == length(max_etas) == length(min_etas)

    iterations = reshape(iterations, length(gamma_factors), length(eta_ratios))
    gammas = reshape(gammas, length(gamma_factors), length(eta_ratios))

    iterations[iterations .>= maxiter] .= NaN

    return eta_ratios, gamma_factors, iterations, max_etas, min_etas, gammas
end


function create_plot_gamma_factors(etas, gamma_factors, iters; nx=128, logscale=true)
    n_etas = length(etas)
    iters = iters ./ nx

    islg = ""
    if logscale
        gamma_factors = log10.(gamma_factors)
        iters = log10.(iters)
        etas = log10.(etas)
        islg = " (log)"
    end

    colours = resample(ColorSchemes.viridis, 2n_etas + 1)[2:2:2n_etas]
    markers = [:circle, :diamond, :utriangle, :dtriangle, :ltriangle, :rtriangle, :cross, :rect, :star4, :pentagon, :star5, :hexagon, :star6]
    markersize=15

    fig = Figure(size=(700, 800))
    # line plot for each viscosity ratio
    lin = Axis(fig[1,1], xlabel="gamma factor" * islg, ylabel="iterations / nx"  * islg, title="Required Iterations for Several Inclusions")

    for (i, eta) = enumerate(etas)
        scatterlines!(lin, gamma_factors, iters[:, i], color=colours[i], marker=markers[i], markersize=markersize, label=string(eta))
    end
    Legend(fig[1,2], lin, "viscosity ratio"  * islg, orientation=:vertical, framevisible=false, padding=(0, 0, 0, 0))

    hidexdecorations!(lin, ticks=false, grid=false)

    # find optimal gamma for each viscosity ratio
    iters_replace_nan = copy(iters)
    iters_replace_nan[isnan.(iters)] .= Inf
    gamma_opts = [gamma_factors[i] for (i, _) ∈ Tuple.(argmin(iters_replace_nan, dims=1)[1, :])]

    # heatmap of iterations
    htmp = Axis(fig[2,1], xlabel="gamma factor" * islg, ylabel="viscosity ratio" * islg)
    htmp_plot = heatmap!(htmp, gamma_factors, etas, iters, colormap=ColorSchemes.viridis)
    Colorbar(fig[2,2], htmp_plot, label="iterations / nx" * islg)

    linkxaxes!(lin, htmp)


    scatterlines!(htmp, gamma_opts, etas, color=:red, marker=:circle, markersize=markersize, label="min iterations")

    return fig

end


function create_plot_gamma_values(etas, gammas, iters, nx=128, logscale=true)
    n_etas = length(etas)
    iters = iters ./ nx

    islg = ""
    if logscale
        gammas = log10.(gammas)
        iters = log10.(iters)
        etas = log10.(etas)
        islg = " (log)"
    end

    colours = resample(ColorSchemes.viridis, 2n_etas + 1)[2:2:2n_etas]
    markers = [:circle, :diamond, :utriangle, :dtriangle, :ltriangle, :rtriangle, :cross, :rect, :star4, :pentagon, :star5, :hexagon, :star6] 
    markersize=15

    fig = Figure(size=(700, 800))
    lin = Axis(fig[1,1], xlabel="gamma" * islg, ylabel="iterations / nx"  * islg, title="Required Iterations for Several Inclusions")
    for (i, eta) = enumerate(etas)
        scatterlines!(lin, gammas[:, i], iters[:, i], color=colours[i], marker=markers[i], markersize=markersize, label=string(eta))
    end
    Legend(fig[1,2], lin, "viscosity ratio" * islg, orientation=:vertical, framevisible=false, padding=(0, 0, 0, 0))

    hidexdecorations!(lin, ticks=false, grid=false)

    htmp = Axis(fig[2,1], xlabel="gamma" * islg, ylabel="viscosity ratio" * islg)
    htmp_plot = scatter!(htmp, reshape(gammas, :), repeat(etas, inner=size(gammas, 1)), color=reshape(iters, :), markersize=47, marker=:rect)
    Colorbar(fig[2,2], htmp_plot, label="iterations / nx" * islg)

    linkxaxes!(lin, htmp)


    # find optimal gamma for each viscosity ratio
    iters_replace_nan = copy(iters)
    iters_replace_nan[isnan.(iters)] .= Inf
    gamma_opts = [gammas[I] for I ∈ argmin(iters_replace_nan, dims=1)[1, :]]
    scatterlines!(htmp, gamma_opts, etas, color=:red, marker=:circle, markersize=markersize, label="min iterations")

    return fig
end


function colapse_lines(etas, gammas, iters, b=1, nx=128, logscale=true)
    iters = iters ./ nx

    islg = ""
    if logscale
        gammas = log10.(gammas)
        iters = log10.(iters)
        etas = log10.(etas)
        islg = " (log)"
    end

    fig = Figure(size=(700, 400))
    ax = Axis(fig[1,1], xlabel="gamma" * islg, ylabel="η^$b iters / nx" * islg)
    for (i, eta) = enumerate(etas)
        scatterlines!(ax, gammas[:, i], iters[:, i] .+ (b * eta))
    end

    return fig
end