using KernelAbstractions
using Enzyme
using CUDA
using JLD2
using CairoMakie

include("../src/tuple_manip.jl")
include("../src/kernels_2D.jl")


@kernel function compute_error!(Er, V, Vex)
    i, j = @index(Global, NTuple)
    if i <= size(V.xc, 1) && j <= size(V.xc, 2)
        Er.xc[i, j] = V.xc[i, j] - Vex.xc[i, j]
    end
    if i <= size(V.yc, 1) && j <= size(V.yc, 2)
        Er.yc[i, j] = V.yc[i, j] - Vex.yc[i, j]
    end
    if i <= size(V.xv, 1) && j <= size(V.xv, 2)
        Er.xv[i, j] = V.xv[i, j] - Vex.xv[i, j]
    end
    if i <= size(V.yv, 1) && j <= size(V.yv, 2)
        Er.yv[i, j] = V.yv[i, j] - Vex.yv[i, j]
    end
end


function run_manufactured_solution(;n=100, γ_factor=1e5,
                                    niter=1e6, freq_recompute=100,
                                    ϵ_ph=1e-8, ϵ_newton=1e-8, ϵ_cg=1e-10,
                                    backend=CPU(), workgroup=64, verbose=false, save=false)
    # physics
    B_val = 1.0
    n_exp = 0.5
    q     = 1 + 1/n_exp
    L     = 1.0

    # manufactured solution
    Vx(x, y) =  sinpi(x) * cospi(y)
    Vy(x, y) = -cospi(x) * sinpi(y)
    fx(x, y) = -(B_val * π^q / 2 / n_exp) * (sinpi(x) * cospi(y)) * (cospi(x) * cospi(y))^(q-2)
    fy(x, y) =  (B_val * π^q / 2 / n_exp) * (cospi(x) * sinpi(y)) * (cospi(x) * cospi(y))^(q-2)
    # numerics
    ϵ̇_bg = eps()
    nx  = ny  = n
    Δx  = Δy  = L/n
    iΔx = iΔy = n/L
    xc  = yc  = LinRange(Δx/2, L - Δx/2, n)
    xv  = yv  = LinRange(0.0, L, n+1)
    # field creation
    P    = (c=KernelAbstractions.zeros(backend, Float64, nx, ny),
            v=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    P̂    = deepcopy(P)
    divV = deepcopy(P)  # velocity divergence
    B    = deepcopy(P)  # prefactor of constituitive relation
    τ    = (c=(xx=KernelAbstractions.zeros(backend, Float64, nx  , ny  ),
               yy=KernelAbstractions.zeros(backend, Float64, nx  , ny  ),
               xy=KernelAbstractions.zeros(backend, Float64, nx+2, ny+2)),
            v=(xx=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1)))  # deviatoric stress tensor
    τ̄    = deepcopy(τ)
    τ̂    = deepcopy(τ)
    V    = (xc=KernelAbstractions.zeros(backend, Float64, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, Float64, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, Float64, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, Float64, nx+1, ny+2))
    dV   = deepcopy(V)  # velocity updates in Newton iteration
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # nonlinear Residual
    K    = deepcopy(V)  # Residuals in CG
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by some vector (used for autodiff)
    R̂    = deepcopy(V)
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    f    = deepcopy(V)  # body force
    Vex  = deepcopy(V)  # exact solution

    # field initialisation
    @views begin
        tplFill!(B, B_val)
        copyto!(f.xc, [fx(x,y) for x=xv, y=yc])
        copyto!(f.yc, [fy(x,y) for x=xc, y=yv])
        copyto!(f.xv[2:end-1, :], [fx(x,y) for x=xc, y=yv])
        copyto!(f.yv[:, 2:end-1], [fy(x,y) for x=xv, y=yc])
        copyto!(Vex.xc, [Vx(x,y) for x=xv, y=yc])
        copyto!(Vex.yc, [Vy(x,y) for x=xc, y=yv])
        copyto!(Vex.xv[2:end-1, :], [Vx(x,y) for x=xc, y=yv])
        copyto!(Vex.yv[:, 2:end-1], [Vy(x,y) for x=xv, y=yc])
    end
    # to test how solver behaves
    tplSet!(V, Vex)
    # residual norms for monitoring convergence
    δ = Inf # CG
    μ = Inf # CG
    χ = Inf # Newton
    ν = Inf # PH

    errV  = Float64[]  # velocity error
    errP  = Float64[]  # pressure error
    iters = Int64[]  # iteration count

    γ    = γ_factor
    # create Kernels
    up_D!      = update_D!(backend, workgroup, (nx+2, ny+2))
    up_dV!     = update_V!(backend, workgroup, (nx+2, ny+2))
    comp_divV! = compute_divV!(backend, workgroup, (nx+1, ny+1))
    comp_P_τ!  = compute_P_τ!(backend, workgroup, (nx+1, ny+1))
    comp_R!    = compute_R!(backend, workgroup, (nx+2, ny+2))
    comp_K!    = compute_K!(backend, workgroup, (nx+2, ny+2))
    up_K!      = update_K!(backend, workgroup, (nx+2, ny+2))
    step_V!    = try_step_V!(backend, workgroup, (nx+2, ny+2))
    init_D!    = initialise_D!(backend, workgroup, (nx+2, ny+2))
    set!       = assign_flux_field!(backend, workgroup, (nx+2, ny+2))
    set_one!   = set_part_to_ones!(backend, workgroup, (nx+2, ny+2))
    set_part!  = assign_part!(backend, workgroup, (nx+2, ny+2))
    inv!       = invert!(backend, workgroup, (nx+2, ny+2))
    comp_err!  = compute_error!(backend, workgroup, (nx+2, ny+2))

    # create function for jacobian-vector product
    
    # compute Dv r * V̄ as Dp r * Dv p * V̄ + Dτ r * Dv τ * V̄
    function jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        make_zero!(Q)
        make_zero!(P̄)
        make_zero!(τ̄)
        tplSet!(P̂, P)
        tplSet!(τ̂.c, τ.c)
        tplSet!(τ̂.v, τ.v)

        autodiff(Forward, comp_P_τ!,
                 Duplicated(P̂, P̄), Duplicated(τ̂, τ̄),
                 Const(P₀),
                 Duplicated(V, V̄),
                 Const(B), Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R!,
                 DuplicatedNoNeed(R̂, Q),
                 Duplicated(P̂, P̄), Duplicated(τ̂, τ̄),
                 Const(f), Const(iΔx), Const(iΔy))
        return nothing
    end

    # function to compute the preconditioner
    # overwrites invM, V̄, Q
    function initialise_invM!(invM, R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        for (i, I) = enumerate(eachindex(invM))
            set_one!(V̄, true, i)
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            set_part!(invM[I], Q[I], true)

            set_one!(V̄, false, i)
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            set_part!(invM[I], Q[I], false)
        end
        inv!(invM)
        return nothing
    end

    println("start computation")
    # Powell Hestenes
    it = 0
    while it < niter && ν > ϵ_ph
        # p_0 = p
        tplSet!(P₀, P)

        # r = f - div τ + grad p - grad( div v)
        comp_P_τ!(P, τ, P₀, V, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        comp_R!(R, P, τ, f, iΔx, iΔy)

        χ = tplNorm(R, 1)
        comp_divV!(divV, V, iΔx, iΔy)
        display(heatmap(divV.v))
        χ₀ = χ
        # Newton iteration
        while it < niter && χ > ϵ_newton * χ₀
            # initialise preconditioner
            # inv(M) = inv(diag (Dv r))
            initialise_invM!(invM, R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            # reference for cg residual
            μ_ref = tplDot(R, R, invM)
            # iteration zero
            # compute residual for CG,
            # k = r - Dv r * dv
            set!(V̄, dV)
            # use R̄ instead of R as first argument because it might get overwritten in autodiff
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            comp_K!(K, R, Q)

            # d = inv(M) * k
            init_D!(D, K, invM)
            μ = tplDot(K, D)
            # start iteration
            it_cg = 1
            while it <= niter && μ > ϵ_cg^2 * μ_ref
                # compute α
                # α = k^T * inv(M) * k / (d^T * Dv r * d)
                set!(V̄, D)
                jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
                α = μ / tplDot(D, Q)

                # dv += α d
                up_dV!(dV, D, α)

                # recompute residual
                if it_cg % freq_recompute == 0
                    # k = r - Dv r * dv
                    set!(V̄, dV)
                    jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
                    comp_K!(K, R, Q)
                else
                    # k = k - α Dv r * d
                    up_K!(K, Q, α)
                end


                # μ = k^T inv(M) k
                μ_new = tplDot(K, K, invM)
                β = μ_new / μ
                μ = μ_new
                # d = β d + inv(M) * k 
                up_D!(D, K, invM, β)

                # periodically check for stagnation
                if it_cg % freq_recompute == 0
                    δ = α * tplNorm(D) / tplNorm(dV) 
                    if δ < ϵ_cg
                        break
                    end
                end
                it_cg += 1
                it    += 1
            end
            #  newton iteration
            λ = .3
            step_V!(V̄, V, dV, λ)
            tplSet!(V, V̄)
            comp_P_τ!(P, τ, P₀, V̄, B, q, ϵ̇_bg, iΔx, iΔy, γ)
            comp_R!(R, P, τ, f, iΔx, iΔy)
            χ = tplNorm(R, Inf)
            verbose && println(round(Int, it / nx), "nx\t", "Newton update, residual = ", χ / χ₀)
        end    
        comp_divV!(divV, V, iΔx, iΔy)
        ν = tplNorm(divV, Inf) # no reference, should be 0
        verbose && println("Pressure update, residual = ", ν)
        # true erros
        comp_err!(V̄, V, Vex)
        push!(errV, tplNorm(V̄, Inf))
        push!(errP, tplNorm(P, Inf))
        push!(iters, it)
    end
    if ν > ϵ_ph
        println("computation did not converge")
    end
    println("computation finished")
    Pc  = Array(P.c)
    Vxc = Array(V.xc)
    Vyc = Array(V.yc)
    Vxe = Array(Vex.xc)
    Vye = Array(Vex.yc)
    save && jldsave("manufactured_raw_data_$(nx)x$(ny).jld2";Pc, Vxc, Vyc, Vxe, Vye, errV, errP, iters)

    return errV, errP, iters
end

run_manufactured_solution(n=100, niter=100)