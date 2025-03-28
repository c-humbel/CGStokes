using Chairmarks
using Statistics
using DelimitedFiles
using CairoMakie
using ColorSchemes

using Random
using KernelAbstractions
using Enzyme
using CUDA

include("../../src/tuple_manip.jl")
include("../../src/kernels_2D.jl")


function measure_residual(n; backend=CPU(), workgroup=64, type=Float64, volume_fractions=false, seed=1234)
    rng = MersenneTwister(seed)
    nx = ny = n
    Lx = Ly = 1
    Δx,  Δy  = Lx / nx, Ly / ny
    iΔx, iΔy = inv(Δx), inv(Δy)

    ϵ̇_bg = eps()
    q = 1. + 1/3  
    γ = 1.

    xc = LinRange(-0.5Lx + 0.5Δx, 0.5Lx - 0.5Δx, nx)
    yc = LinRange(-0.5Ly + 0.5Δy, 0.5Ly - 0.5Δy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)

    # field initialisation
    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny),
            v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    B    = deepcopy(P)  # prefactor of constituitive relation
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx+2, ny+2)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))  # deviatoric stress tensor
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    R    = deepcopy(V)  # Residual
    f    = deepcopy(V)  # body force

    tplFill!(B, 1)

    for a = V
        copyto!(a, rand(rng, size(a)...))
    end
    
    if volume_fractions
        xo = 0.0Lx
        yo = 1.2Lx
        rf = 1.0Lx
        rb = 1.6Lx
        ωₐ = (c=KernelAbstractions.zeros(backend, Float64, nx+2, ny+2),
              v=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1),
              xc=KernelAbstractions.zeros(backend, Float64, nx+1, ny),
              yc=KernelAbstractions.zeros(backend, Float64, nx, ny+1),
              xv=KernelAbstractions.zeros(backend, Float64, nx+2, ny+1),
              yv=KernelAbstractions.zeros(backend, Float64, nx+1, ny+2))
        ωₛ = deepcopy(ωₐ)

        initialise_volume_fractions_ring_segment!(ωₐ, ωₛ, xo, yo, rf, rb,  xc, yc, xv, yv)
        comp_P_τ!(_P, _τ, _P₀, _V, _B, _q, _ϵ̇_bg, _iΔx, _iΔy, _γ) = compute_P_τ_weighted!(backend, workgroup, (nx+1, ny+1))(_P, _τ, _P₀, _V, _B, _q, ωₐ, ωₛ, _ϵ̇_bg, _iΔx, _iΔy, _γ)
        comp_R!(_R, _P, _τ, _f, _iΔx, _iΔy)    = compute_R_weighted!(backend, workgroup, (nx+2, ny+2))(_R, _P, _τ, _f, ωₐ, ωₛ, _iΔx, _iΔy)
    else
        comp_P_τ!  = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
        comp_R!    = compute_R!(backend, workgroup, (nx+2, ny+2))
    end

    # measure time to compute residual
    res = @be begin
        comp_P_τ!(P, τ, P₀, V, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        comp_R!(R, P, τ, f, iΔx, iΔy)
        KernelAbstractions.synchronize(backend)
    end evals=5 samples=1000

    timings = [s.time for s in res.samples]

    # effective throughput (GB / s)
    # ideally, read:  B (2 arrays), V (4 arrays) P₀ (2 arrays), f (4 arrays)
    #          write: R (4 arrays)
    # currently:
    #   Pressure and stress: 8 reads, 8 writes, ignoring BC
    #   Residual: 12 reads, 4 writes
    if volume_fractions
        # comp_P_τ reads velocity ωₛ, comp_R reads cell ωₛ, all of ωₐ
        A_eff = (2*16 + 4 + 8) * nx * ny * sizeof(type) / 1e9
    else
        A_eff = (2*16) * nx * ny * sizeof(type) / 1e9
    end

    return A_eff, median(timings), quantile(timings, 0.05), quantile(timings, 0.95)

end


function measure_jvp(n; backend=CPU(), workgroup=64, type=Float64, volume_fractions=false, seed=1234)
    rng = MersenneTwister(seed)

    nx, ny = n, n
    Lx = Ly = 1
    Δx,  Δy  = Lx / nx, Ly / ny
    iΔx, iΔy = inv(Δx), inv(Δy)

    γ    = 1.
    q    = 1.33
    ϵ̇_bg = eps()


    xc = LinRange(-0.5Lx + 0.5Δx, 0.5Lx - 0.5Δx, nx)
    yc = LinRange(-0.5Ly + 0.5Δy, 0.5Ly - 0.5Δy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)

    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny), v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    B    = deepcopy(P)
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx+2, ny+2)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))
    dτ   = deepcopy(τ)
    P₀   = deepcopy(P)  # old pressure
    dP   = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny  ),
            yc=KernelAbstractions.zeros(backend, type, nx  , ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    dV   = deepcopy(V) 
    R    = deepcopy(V)  # Residuals
    Q    = deepcopy(V)  # row of Jacobian of compute_R wrt. V
    f    = deepcopy(V)

    tplFill!(B, 1)

    for a = V
        copyto!(a, rand(rng, size(a)...))
    end

    xo = 0.0Lx
    yo = 1.2Lx
    rf = 1.0Lx
    rb = 1.6Lx
    ωₐ = (c=KernelAbstractions.zeros(backend, Float64, nx+2, ny+2),
            v=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1),
            xc=KernelAbstractions.zeros(backend, Float64, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, Float64, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, Float64, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, Float64, nx+1, ny+2))
    ωₛ = deepcopy(ωₐ)

    initialise_volume_fractions_ring_segment!(ωₐ, ωₛ, xo, yo, rf, rb,  xc, yc, xv, yv)

    comp_P_τ_vf! = compute_P_τ_weighted!(backend, workgroup, (nx+1, ny+1))
    comp_R_vf!   = compute_R_weighted!(backend, workgroup, (nx+2, ny+2))


    function jvp_R_vf(R, Q, P, P̄, τ, τ̄, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
        autodiff(Forward, comp_P_τ_vf!, DuplicatedNoNeed(P, P̄), DuplicatedNoNeed(τ, τ̄),
                    Const(P₀), Duplicated(V, V̄), Const(B),
                    Const(q), Const(ωₐ), Const(ωₛ), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R_vf!, DuplicatedNoNeed(R, Q),
                    Duplicated(P, P̄), Duplicated(τ, τ̄), Const(f), Const(ωₐ), Const(ωₛ), Const(iΔx), Const(iΔy))
        return nothing
    end

    comp_P_τ!  = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
    comp_R!    = compute_R!(backend, workgroup, (nx+2, ny+2))

    function jvp_R(R, Q, P, P̄, τ, τ̄, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        autodiff(Forward, comp_P_τ!, DuplicatedNoNeed(P, P̄), DuplicatedNoNeed(τ, τ̄),
                    Const(P₀), Duplicated(V, V̄), Const(B),
                    Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R!, DuplicatedNoNeed(R, Q),
                    Duplicated(P, P̄), Duplicated(τ, τ̄), Const(f), Const(iΔx), Const(iΔy))
        return nothing
    end

    if volume_fractions
        res = @be begin
            jvp_R_vf(R, Q, P, dP, τ, dτ, V, dV, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            KernelAbstractions.synchronize(backend)
        end evals=5 samples=1000
    else
        res = @be begin
            jvp_R(R, Q, P, dP, τ, dτ, V, dV, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            KernelAbstractions.synchronize(backend)
        end evals=5 samples=1000
    end

    timings = [s.time for s in res.samples]

    # effective throughput (GB / s)
    # ideally, read: V, V̄, write Q 
    # what actually happens, no idea
    # but assume all arrays are treated as in direct computation, and shadows are read and written
    A_eff = (volume_fractions ? 2*16 + 12 + 8 : 2*16 + 8) * nx * ny * sizeof(type) / 1e9

    return  A_eff, median(timings), quantile(timings, 0.05), quantile(timings, 0.95)
    return nothing
end


function benchmark(function_name, ns=2 .^(5:8); save=false, backend=CPU(), workgroup=64)
    results = zeros(length(ns), 9)
    if function_name == "residual"
        f = measure_residual
    elseif function_name == "jvp"
        f = measure_jvp
    else
        return Float64[]
    end

    for (i, n) = enumerate(ns)
        println("start resolution ", n)
        out_vf = f(n; backend=backend, workgroup=workgroup, type=Float64, volume_fractions=true)
        out = f(n; backend=backend, workgroup=workgroup, type=Float64, volume_fractions=false)
        results[i, :] .= n, out..., out_vf...
    end
    save && writedlm("results_$function_name.csv", results, ',')
    return results
end



function create_throughput_plot(results; title="Residual", savefig=false)
    ns = results[:, 1]

    with_theme(theme_latexfonts()) do
        violet, green = resample(ColorSchemes.viridis, 5)[[1, 3]]
        fig = Figure(fontsize=16,size=(600,400))
        ax  = Axis(fig[1,1], title="Memory Throughput of $title Computation",
                   xlabel=L"n", ylabel=L"T_{eff} \; (\text{GB/s})",
                   xscale=log2, xticks=ns)
        hlines!(ax, 1370, linestyle=:dash, color=:grey)
        for (with_vf, offset, colour, marker) = zip(["without", "with"], [2, 6], [violet, green], [:circle, :diamond])
            median_throughput = results[:, offset] ./ results[:, offset+1]
            upper_throughput = results[:, offset] ./ results[:, offset+2]
            lower_throughput = results[:, offset] ./ results[:, offset+3]
            rangebars!(ax, ns, lower_throughput, upper_throughput, color=colour, whiskerwidth=10,)
            scatter!(ax, ns, median_throughput, color=colour, markersize=15, marker=marker, label=with_vf * " volume fractions")
        end
        Legend(fig[2, 1], ax, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))
        if savefig
            save("throughput_$(lowercase(title)).pdf", fig)
        else
            display(fig)
        end
    end
end


res = benchmark("jvp", 2 .^(8:13), backend=CUDABackend(), workgroup=(32, 8), save=true)
create_throughput_plot(res, title="Jacobian Vector Product", savefig=true)

