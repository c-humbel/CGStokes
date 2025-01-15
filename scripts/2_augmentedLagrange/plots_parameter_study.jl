using CairoMakie
using ColorSchemes
using Printf

function load_data(filename, maxiter=5e5)
    viscosities = []
    gammas = []
    iterations = []

    for (i, line) ∈ enumerate(eachline(filename))
        if i == 1 continue end
        η_ratio, γ_factor, it = split(line, ", ")
        push!(viscosities, parse(Float64, η_ratio))
        push!(gammas, parse(Float64, γ_factor))
        push!(iterations, parse(Int, it))
    end

    unique!(viscosities)
    unique!(gammas)

    iterations = reshape(iterations, length(gammas), length(viscosities))

    iterations[iterations .>= maxiter] .= NaN

    return viscosities, gammas, iterations
end


function create_plot(etas, gammas, iters; nx=128)
    n_etas = length(etas)

    colours = resample(ColorSchemes.viridis, 2n_etas + 1)[2:2:2n_etas]
    markers = [:circle, :diamond, :utriangle, :dtriangle, :ltriangle, :rtriangle, :cross, :rect, :star4, :pentagon, :star5, :hexagon, :star6] 

    fig = Figure(size=(700, 800));
    # line plot for each viscosity ratio
    lin = Axis(fig[1,1][1,1], xlabel="gamma factor (log)", ylabel="iterations / nx (log)", title="Required Iterations for Several Inclusions");

    for (i, eta) = enumerate(etas)
        scatterlines!(lin, log10.(gammas), log10.(iters[:, i] ./ nx), color=colours[i], marker=markers[i], label=@sprintf("%.e", eta))
    end
    Legend(fig[1,1][1,2], lin, "viscosity ratio", orientation=:vertical, framevisible=false, padding=(0, 0, 0, 0))

    # find optimal gamma for each viscosity ratio
    iters_replace_nan = copy(iters)
    iters_replace_nan[isnan.(iters)] .= Inf
    gamma_opts = [gammas[i] for (i, _) ∈ Tuple.(argmin(iters_replace_nan, dims=1)[1, :])]

    # heatmap of iterations
    htmp = Axis(fig[2,1][1,1], xlabel="gamma (log)", ylabel="viscosity ratio (log)")
    htmp_plot = heatmap!(htmp, log10.(gammas), log10.(etas), log10.(iters ./ nx), colormap=ColorSchemes.viridis)
    Colorbar(fig[2,1][1,2], htmp_plot, label="iterations / nx (log)")

    scatterlines!(htmp, log10.(gamma_opts), log10.(etas), color=:black, marker=:cross, label="min iterations")

    return fig

end




