using CairoMakie
using ColorSchemes
using Enzyme
using KernelAbstractions
using CUDA
using DelimitedFiles
using Interpolations

include("../../src/tuple_manip.jl")
include("../4_nonlinear_augmLagrange/kernels_free_slip.jl")
include("../4_nonlinear_augmLagrange/kernels_volume_fractions.jl")


function setup_arolla(nx::Int, aspect_ratio, filepath, backend)
    # data from https://frank.pattyn.web.ulb.be/ismip/welcome.html#Input
    data, _ = readdlm(filepath, '\t', Float64, header=true)
    x_max = maximum(data[:, 1])
    x_min = minimum(data[:, 1])
    Lx    = x_max - x_min
    y_max = maximum(data[:, 2:3])
    y_min = minimum(data[:, 2:3])
    Ly    = y_max - y_min

    # grid spacing, centers should be at x coordinates from data
    Δx = Lx / (nx - 1)
    ny = round(Int, Ly / (Δx * aspect_ratio)) + 1
    Δy = Ly / (ny - 1)
    # coodrindates of grid centers & vertices
    xc = LinRange(x_min, x_max, nx)
    yc = LinRange(y_min, y_max, ny)
    xv = LinRange(x_min - 0.5Δx, x_max + 0.5Δx, nx+1)
    yv = LinRange(y_min - 0.5Δy, y_max + 0.5Δy, ny+1)


    ωₐ = (c=KernelAbstractions.zeros(backend, Float64, nx+2, ny+2),
           v=KernelAbstractions.zeros(backend, Float64, nx+1, ny+1),
           xc=KernelAbstractions.zeros(backend, Float64, nx+1, ny),
           yc=KernelAbstractions.zeros(backend, Float64, nx, ny+1),
           xv=KernelAbstractions.zeros(backend, Float64, nx+2, ny+1),
           yv=KernelAbstractions.zeros(backend, Float64, nx+1, ny+2))
    ωₛ = deepcopy(ωₐ)

    # compute volume fractions
    fₐ = extrapolate(interpolate!((data[:, 1],), data[:, 3], Gridded(Linear())), Flat())
    fₛ = extrapolate(interpolate!((data[:, 1],), data[:, 2], Gridded(Linear())), Flat())
    initialise_volume_fractions_from_function!(ωₐ, fₐ, xc, yc, xv, yv, 1., 0.)
    initialise_volume_fractions_from_function!(ωₛ, fₛ, xc, yc, xv, yv, 0., 1.)
    return Lx, Ly, nx, ny, Δx, Δy, xc, yc, xv, yv, ωₐ, ωₛ
end


    

function run(filepath; n=126, niter=10000, γ_factor=1., aspect=0.5,
            ϵ_cg=1e-3, ϵ_ph=1e-6, ϵ_newton=1e-3, freq_recompute=100,
            backend=CPU(), workgroup=64, Float64=Float64, verbose=false)

    # parameters from ISMIP, transformation according to Schoof (2006)
    ρgy   = 9.81 * 910.
    A_val = 1e-16 / 31556926.
    n_exp = 3.
    B_val = A_val^(-1/n_exp)
    q     = 1. + 1/n_exp 

    # numerical parameters
    ϵ̇_bg = eps()
    γ    = γ_factor

   
    # setup test case
    Lx, Ly, nx, ny, Δx, Δy, xc, yc, xv, yv, ωₐ, ωₛ = setup_arolla(n, aspect, filepath, backend)

    iΔx, iΔy = inv(Δx), inv(Δy)

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

    #  field initialisation
    tplFill!(B, B_val)
    fill!(f.yc, ρgy)
    fill!(f.yv, ρgy)

    # residual norms for monitoring convergence
    δ = Inf # CG
    μ = Inf # CG
    χ = Inf # Newton
    ν = Inf # Pressure
    
    χ_ref = tplNorm(f)

    # visualisation
    itercounts = []
    res_newton = []

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

    # start timer
    verbose && println("start computation")
    t_init = Base.time()
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
                    println("CG residual, μ = ", μ / μ_ref, ", δ = ", δ)

                end

                # periodically check for stagnation
                if it_cg % freq_recompute == 0
                    δ = α * tplNorm(D) / tplNorm(dV) 
                    if δ < ϵ_cg
                        break
                    end
                end
            end
            # damped to newton iteration
            # find λ st. r(v - λ dv) < r(v)
            λ = .5
            step_V!(V̄, V, dV, λ)
            comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
            χ_new = tplNorm(R) / χ_ref
            # while χ_new >= χ && λ > 1e-3
            #     λ /= MathConstants.golden
            #     step_V!(V̄, V, dV, λ)
            #     comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            #     comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
            #     χ_new = tplNorm(R) / χ_ref
            # end
            tplSet!(V, V̄)
            χ = χ_new
 
            push!(itercounts, it)
            push!(res_newton, χ)
            
            verbose && println("Newton residual = ", χ, "; λ = ", λ, "; total iteration count: ", round(Int, it / nx), "nx")
        end    
        comp_divV!(divV, V,  ωₐ, ωₛ, iΔx, iΔy)
        ν = tplNorm(divV) / tplNorm(P)
        verbose && println("Pressure residual = ", ν, ", Newton residual = ", χ, ", CG residual = ", δ)
    end
    Δt = Base.time() - t_init
    verbose && println("Time elapsed: ", Δt, "s")
    return P, V, ωₐ, ωₛ, xc, yc, itercounts, res_newton
end

n = 254
P, V, ωₐ, ωₛ, xc, yc, itercounts, res_newton = run("../Examples/ismip_arolla_100.txt";
                                                  n=n, aspect=0.3,
                                                  γ_factor=1e8, niter=500n,
                                                  ϵ_ph=1e-8, ϵ_cg=1e-10, ϵ_newton=1e-8,
                                                  freq_recompute=100, verbose=false);


@views function surface_velocity(ωac, ωsc, Vm)
# dimensions: ω = (nx+2,ny+2), Vm =(nx,ny)
    V_surf = zeros(size(Vm, 1))

    for i = axes(ωac, 1)[2: end-1]
        j = findfirst(w -> w == 0, ωac[i, :])
        if !isnothing(j) && ωsc[i, j] != 0
            V_surf[i-1] = Vm[i-1, max(1,j-3)]
        else
            V_surf[i-1] = NaN
        end
    end

    return V_surf
end

Pc = Array(P.c)
Vxc = Array(V.xc)
Vyc = Array(V.yc)
ωsc = Array(ωₛ.c)
ωac = Array(ωₐ.c)
Vm  = 0.5 .* sqrt.((Vxc[2:end, :] .+ Vxc[1:end-1, :]) .^2 .+ (Vyc[:, 2:end] .+ Vyc[:, 1:end-1]) .^2)
Vs = surface_velocity(ωac, ωsc, Vm)
# visualisation
fig = Figure(size=(800,1200))
axs = (ω =Axis(fig[1,1][1,1], title="domain", xlabel="x", ylabel="y"),
       Pc=Axis(fig[2,1][1,1], title="pressure", xlabel="x", ylabel="y"),
       Vm=Axis(fig[3,1][1,1], title="velocity magnitude", xlabel="x", ylabel="y"),
       Vs=Axis(fig[4,1][1,1], title="surface velocity", xlabel="x", ylabel="m/s"),
       Er=Axis(fig[5,1][1,1], title="Convergence of Newton Method", xlabel="iterations / nx", ylabel="residual norm")
       )

plt = (ω =heatmap!(axs.ω , ωac .* ωsc, colormap=:greys),
       Pc=heatmap!(axs.Pc, Pc, colormap=ColorSchemes.viridis),
       Vm=heatmap!(axs.Vm, Vm, colormap=ColorSchemes.viridis),
       Vs=scatter!(axs.Vs, xc, Vs, color=:blue),
       Er=scatterlines!(axs.Er, itercounts ./ size(Pc, 1), log10.(res_newton), color=:purple))

cbar= (ω =Colorbar(fig[1,1][1, 2], plt.ω),
       Pc=Colorbar(fig[2,1][1, 2], plt.Pc),
       Vm=Colorbar(fig[3,1][1, 2], plt.Vm),
       )

display(fig)


