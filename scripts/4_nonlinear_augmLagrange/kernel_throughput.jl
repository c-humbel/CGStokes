using Chairmarks
using Statistics

using Random
using KernelAbstractions
using Enzyme
using CUDA

include("../../src/tuple_manip.jl")
include("kernels_free_slip.jl")
include("init_many_inclusions.jl")


function measure_residual(n; backend=CPU(), workgroup=64, type=Float64)
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
               xy=KernelAbstractions.zeros(backend, type, nx, ny)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))  # deviatoric stress tensor
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    R    = deepcopy(V)  # Residual
    f    = deepcopy(V)  # body force

    initialise_B_f!(B, f, 1., 1., 2., xc, yc, xv, yv, Lx, Ly; ninc=1)

    comp_P_τ!  = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
    comp_R!    = compute_R!(backend, workgroup, (nx+2, ny+2))

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
    A_eff = (2*16) * nx * ny * sizeof(type) / 1e9

    return A_eff ./ (median(timings), quantile(timings, 0.05), quantile(timings, 0.95))

end


function measure_jvp(n; backend=CPU(), workgroup=64, type=Float64, seed=1234)
    rng = MersenneTwister(seed)

    nx, ny = n, n
    Lx = Ly = 1
    Δx,  Δy  = Lx / nx, Ly / ny
    iΔx, iΔy = inv(Δx), inv(Δy)

    γ    = 1.
    q    = 1.33
    ϵ̇_bg = eps()

    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny), v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    B    = deepcopy(P)
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx, ny)),
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

    comp_P_τ!  = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
    comp_R!    = compute_R!(backend, workgroup, (nx+2, ny+2))

    function jvp_R(R, Q, P, P̄, τ, τ̄, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        autodiff(Forward, comp_P_τ!, DuplicatedNoNeed(P, P̄), DuplicatedNoNeed(τ, τ̄),
                 Const(P₀), Duplicated(V, V̄), Const(B),
                 Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R!, DuplicatedNoNeed(R, Q),
                 Duplicated(P, P̄), Duplicated(τ, τ̄), Const(f), Const(iΔx), Const(iΔy))
    end

    res = @be begin
        jvp_R(R, Q, P, dP, τ, dτ, V, dV, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        KernelAbstractions.synchronize(backend)
    end evals=5 samples=1000

    timings = [s.time for s in res.samples]

    # effective throughput (GB / s)
    # ideally, read: V, V̄, write Q 
    # what actually happens, no idea
    # A_eff = (12) *nx * ny * sizeof(type) / 1e9

    return #= A_eff ./ =# (median(timings), quantile(timings, 0.05), quantile(timings, 0.95))

end