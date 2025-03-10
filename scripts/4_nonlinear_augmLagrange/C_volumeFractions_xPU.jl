using CairoMakie
using ColorSchemes
using Enzyme
using KernelAbstractions
using Random
using CUDA

include("../../src/tuple_manip.jl")
include("kernels_free_slip.jl")
include("kernels_volume_fractions.jl")

function nonlinear_inclusion(;n=126, niter=10000, γ_factor=1.,
                            ϵ_cg=1e-3, ϵ_ph=1e-6, ϵ_newton=1e-3, freq_recompute=100,
                            backend=CPU(), workgroup=64, type=Float64, verbose=false)
    L_ref = 1. # reference length 
    ρg_b  = 1. # background density

    # parameters are, using n, A from Glen's law
    # q = 1 + 1/n, B = A ^ (-1/n) (Schoof 2006)
    B_avg = 1. 
    q = 1. + 1/3  

    Lx = Ly = L_ref

    rₐ = 1.6Lx
    rₛ = 0.9Ly
    x₀ = 0
    y₀ = -1.2Ly
    ri  = [0.1Lx, 0.15Lx, 0.2Lx]
    xi  = [-0.3Lx, 0.0Lx, 0.2Ly]
    yi  = [-0.3Ly, 0.2Ly, -0.2Ly]
    ρgi = [0.0, 0.0, 0.0]
    nx = ny = n
    ϵ̇_bg = eps()

    Δx,  Δy  = Lx / nx, Ly / ny
    iΔx, iΔy = inv(Δx), inv(Δy)
    xc = LinRange(-0.5Lx + 0.5Δx, 0.5Lx - 0.5Δx, nx)
    yc = LinRange(-0.5Ly + 0.5Δy, 0.5Ly - 0.5Δy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)


    # field initialisation
    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny),
            v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)  # velocity divergence
    B    = deepcopy(P)  # prefactor of constituitive relation
    # ϵ̇_E  = deepcopy(P)  # squared invariant of strain rate
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx  , ny  ),
               yy=KernelAbstractions.zeros(backend, type, nx  , ny  ),
               xy=KernelAbstractions.zeros(backend, type, nx+2, ny+2)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))  # deviatoric stress tensor
    τ̄    = deepcopy(τ)
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    dV   = deepcopy(V)  # velocity updates in Newton iteration
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # nonlinear Residual
    K    = deepcopy(V)  # Residuals in CG
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by some vector (used for autodiff)
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    f    = deepcopy(V)  # body force
    ωₐ   = (c =KernelAbstractions.zeros(backend, type, nx+2, ny+2),
            v =KernelAbstractions.zeros(backend, type, nx+1, ny+1),
            xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
            yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    ωₛ   = deepcopy(ωₐ)

    R̂    = deepcopy(V)
    P̂    = deepcopy(P)
    τ̂    = deepcopy(τ)

    initialise_volume_fractions_ring_segment!(ωₐ, ωₛ,  x₀, y₀, rₐ, rₛ, xc, yc, xv, yv)
    initialise_f!(f, xi, yi, ri, ρgi, ρg_b, xc, yc, xv, yv)

    tplFill!(B, B_avg)

    γ = γ_factor


    # residual norms for monitoring convergence
    δ = Inf # CG
    μ = Inf # CG
    χ = Inf # Newton
    ν = Inf # Pressure
    
    χ_ref = tplNorm(f)

    # visualisation
    itercounts = []
    res_newton = []

    fig = Figure(size=(800,900))
    axs = (ρg=Axis(fig[1,1][1,1], aspect=1, title="ρg"),
           ω =Axis(fig[1,2][1,1], aspect=1, title="volume fractions"),
           Vx=Axis(fig[2,1][1,1], aspect=1, title="horizontal velocity"),
           Vy=Axis(fig[2,2][1,1], aspect=1, title="vertical velocity"),
           Pc=Axis(fig[3,1][1,1], aspect=1, title="pressure"),
           Er=Axis(fig[3,2][1,1], title="Convergence of Newton Method", xlabel="iterations / nx", ylabel="residual norm")
           )

    plt = (ρg=heatmap!(axs.ρg, Array(f.yc), colormap=ColorSchemes.viridis),
           ω =heatmap!(axs.ω , Array(ωₐ.c .* ωₛ.c), colormap=:greys),
           Vx=heatmap!(axs.Vx, Array(V.xc), colormap=ColorSchemes.viridis),
           Vy=heatmap!(axs.Vy, Array(V.yc), colormap=ColorSchemes.viridis),
           Pc=heatmap!(axs.Pc, Array(P.c), colormap=ColorSchemes.viridis),
           )

    cbar= (ρg=Colorbar(fig[1, 1][1, 2], plt.ρg),
           ω =Colorbar(fig[1, 2][1, 2], plt.ω),
           Pc=Colorbar(fig[3, 1][1, 2], plt.Pc),
           Vx=Colorbar(fig[2, 1][1, 2], plt.Vx),
           Vy=Colorbar(fig[2, 2][1, 2], plt.Vy)
           )

    display(fig)

    # create Kernels
    up_D!      = update_D!(backend, workgroup, (nx+2, ny+2))
    up_dV!     = update_V!(backend, workgroup, (nx+2, ny+2))
    comp_divV! = compute_divV_weighted!(backend, workgroup, (nx+1, ny+1))
    comp_P_τ!  = compute_P_τ_weighted!(backend, workgroup, (nx+1, ny+1))
    comp_R!    = compute_R_weighted!(backend, workgroup, (nx+2, ny+2))
    comp_K!    = compute_K!(backend, workgroup, (nx+2, ny+2))
    up_K!      = update_K!(backend, workgroup, (nx+2, ny+2))
    step_V!    = try_step_V!(backend, workgroup, (nx+2, ny+2))
    init_D!    = initialise_D!(backend, workgroup, (nx+2, ny+2))
    set!       = assign_flux_field!(backend, workgroup, (nx+2, ny+2))
    set_one!   = set_part_to_ones!(backend, workgroup, (nx+2, ny+2))
    set_part!  = assign_part!(backend, workgroup, (nx+2, ny+2))
    inv!       = invert!(backend, workgroup, (nx+2, ny+2))

    # create function for jacobian-vector product
    
    # compute Dv r * V̄ as Dp r * Dv p * V̄ + Dτ r * Dv τ * V̄
    function jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
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
                 Const(B), Const(q), Const(ωₐ), Const(ωₛ), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

        autodiff(Forward, comp_R!,
                 DuplicatedNoNeed(R̂, Q),
                 Duplicated(P̂, P̄), Duplicated(τ̂, τ̄),
                 Const(f), Const(ωₐ), Const(ωₛ), Const(iΔx), Const(iΔy))
        return nothing
    end

    # function to compute the preconditioner
    # overwrites invM, V̄, Q
    function initialise_invM!(invM, R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ϵ̇_bg, iΔx, iΔy, γ)
        for (i, I) = enumerate(eachindex(invM))
            set_one!(V̄, true, i)
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            set_part!(invM[I], Q[I], true)

            set_one!(V̄, false, i)
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            set_part!(invM[I], Q[I], false)
        end
        inv!(invM)
        return nothing
    end

    # Powell Hestenes
    it = 0
    while it < niter && ν > ϵ_ph
        # p_0 = p
        tplSet!(P₀, P)

        # r = f - div τ + grad p - grad( div v)
        comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
        comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)

        χ = tplNorm(R) / χ_ref

        # Newton iteration
        while it < niter && χ > ϵ_newton
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
            jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            comp_K!(K, R, Q)

            # d = inv(M) * k
            init_D!(D, K, invM)
            μ = tplDot(K, D)
            # δ = tplNorm(K, Inf) / δ_ref
            # start iteration
            it_cg = 1
            while it <= niter && μ > ϵ_cg^2 * μ_ref
                # compute α
                # α = k^T * inv(M) * k / (d^T * Dv r * d)
                set!(V̄, D)
                jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
                α = μ / tplDot(D, Q)

                # dv += α d
                up_dV!(dV, D, α)

                # recompute residual
                if it_cg % freq_recompute == 0
                    # k = r - Dv r * dv
                    set!(V̄, dV)
                    jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
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

                it_cg += 1
                it += 1

                if verbose && it_cg % n == 0
                    println("CG residual = ", μ / μ_ref)
                    # plt.Pc[3][] .= Array(P.c)
                    # plt.Vx[3][] .= Array(K.xc)
                    # plt.Vy[3][] .= Array(K.yc)
                    # plt.Pc.colorrange[] = (min(-1e-10,minimum(P.c )), max(1e-10,maximum(P.c )))
                    # plt.Vx.colorrange[] = (min(-1e-10,minimum(K.xc)), max(1e-10,maximum(K.xc)))
                    # plt.Vy.colorrange[] = (min(-1e-10,minimum(K.yc)), max(1e-10,maximum(K.yc)))
                    # display(fig)
                end

                # periodically check for stagnation
                if it_cg % freq_recompute == 0
                    δ = α * tplNorm(D) / tplNorm(dV) 
                    if δ < ϵ_cg
                        println("GC relative update: ", δ)
                        break
                    end
                end
            end
            # damped to newton iteration
            # find λ st. r(v - λ dv) < r(v)
            λ = 1.
            step_V!(V̄, V, dV, λ)
            comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
            χ_new = tplNorm(R) / χ_ref
            while χ_new >= χ && λ > 1e-3
                λ /= MathConstants.golden
                step_V!(V̄, V, dV, λ)
                comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
                comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
                χ_new = tplNorm(R) / χ_ref
            end
            tplSet!(V, V̄)
            χ = χ_new
 
            push!(itercounts, it)
            push!(res_newton, χ)

            # update plot
            plt.Pc[3][] .= Array(P.c)
            plt.Vx[3][] .= Array(V.xc)
            plt.Vy[3][] .= Array(V.yc)
            # plt.Sr[3][] .= log10.(Array(ϵ̇_E.c))
            plt.Pc.colorrange[] = (min(-1e-10,minimum(P.c )), max(1e-10,maximum(P.c )))
            plt.Vx.colorrange[] = (min(-1e-10,minimum(V.xc)), max(1e-10,maximum(V.xc)))
            plt.Vy.colorrange[] = (min(-1e-10,minimum(V.yc)), max(1e-10,maximum(V.yc)))
            # plt.Sr.colorrange[] = (min(-1,log10(minimum(ϵ̇_E.c))), max(1,log10(maximum(ϵ̇_E.c))))

            scatterlines!(axs.Er, itercounts ./ nx, log10.(res_newton), color=:purple)

            display(fig)
            
            println("Newton residual = ", χ, "; λ = ", λ, "; total iteration count: ", it)
        end    
        comp_divV!(divV, V,  ωₐ, ωₛ, iΔx, iΔy)
        ν = tplNorm(divV) / tplNorm(P)
        println("Pressure residual = ", ν, ", Newton residual = ", χ, ", CG residual = ", δ)
     end
 
     return it, P, V, R
end

n = 62
nonlinear_inclusion(n=n, γ_factor=1000., niter=1000n, ϵ_ph=1e-5, ϵ_cg=1e-5, ϵ_newton=1e-5, freq_recompute=100, verbose=false);

