using CairoMakie
using ColorSchemes
using Enzyme
using KernelAbstractions
using CUDA
using DelimitedFiles
using Interpolations
using JLD2

include("../../src/tuple_manip.jl")
include("../4_nonlinear_augmLagrange/kernels_free_slip.jl")
include("../4_nonlinear_augmLagrange/kernels_volume_fractions.jl")


function setup_arolla(nx::Int, aspect_ratio, filepath, backend)
    # data from https://frank.pattyn.web.ulb.be/ismip/welcome.html#Input
    data  = readdlm(filepath, '\t', Float64)
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

# post processsing                                                  
function compute_boundary_properties(ωac, ωsc, Vxv, Vyc, τcxy, Pc, Δy)
    ΔP_bed = zeros(size(Pc, 1))
    τxy_bed = zeros(size(τcxy, 1) - 2)
    Vx_surf = zeros(size(Vyc, 1))
    Vy_surf = zeros(size(Vyc, 1))
    for i = eachindex(ΔP_bed)
        j_bed = findfirst(w -> w == 1, ωsc[i+1, :])
        j_air = findlast(w -> w == 1, ωac[i+1, :])
        if !isnothing(j_bed) && !isnothing(j_air) && j_bed <= j_air
            ρgh = 910 * 9.81 * Δy * (j_air - j_bed + 1) / 1000 # kPa
            ΔP_bed[i] = Pc[i, j_bed-1] - ρgh
            τxy_bed[i] = τcxy[i+1, j_bed]
            Vx_surf[i] = Vxv[i+1, j_air-1]
            Vy_surf[i] = Vyc[i, j_air-1]
        else
            ΔP_bed[i] = NaN
            τxy_bed[i] = NaN
            Vx_surf[i] = NaN
            Vy_surf[i] = NaN
        end
    end
    return Vx_surf, Vy_surf, τxy_bed, ΔP_bed
end

function extract_data(P, V, τ, ωₐ, ωₛ, xc, yc, itercounts, residuals)
    nx = length(xc)
    ny = length(yc)
    Pc = Array(P.c) ./ 1000 # kPa
    Vxc = Array(V.xc) .* 31556926 # m/year
    Vxv = Array(V.xv) .* 31556926
    Vyc = Array(V.yc) .* 31556926
    ωsc = Array(ωₛ.c)
    ωac = Array(ωₐ.c)
    τcxy = Array(τ.c.xy) ./ 1000 # kPa

    Vm  = 0.5 .* sqrt.((Vxc[2:end, :] .+ Vxc[1:end-1, :]) .^2 .+ (Vyc[:, 2:end] .+ Vyc[:, 1:end-1]) .^2)

    Vxs, Vys, τxyb, ΔPb = compute_boundary_properties(ωac, ωsc, Vxv, Vyc, τcxy, Pc, yc[2] - yc[1])
    writedlm("compare_arolla_$(nx)x$(ny).dat", hcat(xc ./ 5000, Vxs, Vys, τxyb, ΔPb))


    background = ωac[2:end-1, 2:end-1] .* ωsc[2:end-1, 2:end-1] .== 0
    Pc[background] .= NaN
    Vm[background] .= NaN

    jldsave("raw_data_arolla_$(nx)x$(ny).jld2"; Pc, Vm, Vxc, Vyc, τcxy, ωac, ωsc, xc, yc, itercounts, residuals, compress=true)

    return Pc, Vm, Vxs, Vys, τxyb, ΔPb, xc, yc, itercounts, residuals
end
    
function create_summary_plots(Pc, Vm, Vxs, Vys, τxyb, ΔPb, xc, yc, itercounts, residuals; savefig=true)
    nx = length(xc)
    ny = length(yc)
    with_theme(theme_latexfonts()) do
        violet, green = resample(ColorSchemes.viridis, 5)[[1, 3]]
        fig = Figure(fontsize=16,size=(800,1600))
        axs = (
               Pc=Axis(fig[1,1][1,1], title="Pressure", xlabel="x (m)", ylabel="elevation (m.a.s.l.)"),
               Vm=Axis(fig[2,1][1,1], title="Velocity", xlabel="x (m)", ylabel="elevation (m.a.s.l.)"),
               Vs=Axis(fig[3,1][1,1], title="Surface Velocity", xlabel="x (m)", ylabel="m/a"),
               Tb=Axis(fig[4,1][1,1], title="Basal Shear Stress", xlabel="x (m)", ylabel="kPa"),
               Er=Axis(fig[5,1][1,1], title="Convergence", xlabel="conjugate gradient iterations / nx", ylabel="residual norm")
            )
    
        scatterlines!(axs.Er, itercounts.ph ./ size(Pc, 1), log10.(residuals.ph), color=violet, marker=:diamond, linestyle=:dash, label="Pressure")
        plt = (
              Pc=heatmap!(axs.Pc, xc, yc, Pc, colormap=ColorSchemes.viridis),
              Vm=heatmap!(axs.Vm, xc, yc, Vm, colormap=ColorSchemes.viridis),
              Vs=lines!(axs.Vs, xc, @.(sqrt(Vxs^2 + Vys^2)), color=violet),
              Tb=lines!(axs.Tb, xc, τxyb, color=violet),
              Er=scatterlines!(axs.Er, itercounts.newton ./ size(Pc, 1), log10.(residuals.newton), color=green, marker=:circle, label="Velocity")
              )
    
        cbar= (
              Pc=Colorbar(fig[1,1][1, 2], plt.Pc, label="kPa"),
              Vm=Colorbar(fig[2,1][1, 2], plt.Vm, label="m/a"),
              Er=Legend(fig[5,1][2, 1], axs.Er, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))
            )
    
        if savefig
            save("overview_arolla_$(nx)x$(ny).pdf", fig)
        else
            display(fig)
        end
    end
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
    ν = Inf # PH

    ν_ref = B_val^(-n_exp) * ρgy^n_exp * Lx^n_exp
    χ_ref = Lx * ν_ref

    γ    = γ_factor * B_val * ν_ref^(-2/3)


    # visualisation
    res = (newton=[], ph=[])
    iter = (newton=[], ph=[])

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
    println("start computation")
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
            # damped to newton iteration
            # find λ st. r(v - λ dv) < r(v)
            λ = .3
            step_V!(V̄, V, dV, λ)
            comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
            χ_new = tplNorm(R, Inf) / χ_ref
            # while χ_new >= χ && λ > 1e-3
            #     λ /= MathConstants.golden
            #     step_V!(V̄, V, dV, λ)
            #     comp_P_τ!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
            #     comp_R!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
            #     χ_new = tplNorm(R) / χ_ref
            # end
            tplSet!(V, V̄)
            χ = χ_new
 
            push!(iter.newton, it)
            push!(res.newton, χ)
            
            verbose && println(round(Int, it / nx), "nx\t", "Newton update, residual = ", χ, )
        end    
        comp_divV!(divV, V,  ωₐ, ωₛ, iΔx, iΔy)
        ν = tplNorm(divV, Inf) / ν_ref
        push!(iter.ph, it)
        push!(res.ph, ν)
        verbose && println("Pressure update, residual = ", ν)
    end
    Δt = Base.time() - t_init
    if ν > ϵ_ph
        println("computation did not converge")
    end
    println("computation finished, elapsed time = ", Δt, "s")
    return P, V, τ, ωₐ, ωₛ, xc, yc, iter, res
end

