using CairoMakie
using ColorSchemes
using LinearAlgebra
using Enzyme


function tplNorm(x::NamedTuple, p::Real=2)
    return norm(norm.(values(x), p), p)   
end


function tplDot(x::NamedTuple, y::NamedTuple, a::NamedTuple)
    s = 0.
    for k = keys(x)
        s += dot(x[k], a[k] .* y[k])
    end
    return s
end


function tplDot(x::NamedTuple, y::NamedTuple, a::Real=1.)
    return sum(dot.(values(x), a .* values(y)))
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::NamedTuple)
    for k = keys(dest)
        copyto!(dest[k], a[k] .* src[k])
    end
    return nothing
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::Real=1.)
    copyto!.(values(dest), a .* values(src))
    return nothing
end


function tplScale!(x::NamedTuple, a::Real)
    for k = keys(x)
        x[k] .= a .* x[k]
    end
    return nothing
end


function compute_divV!(divV, V, dx, dy)
    nx, ny = size(divV)
    for j = 1:ny
        for i = 1:nx
            dVx = (V.x[i+1, j] - V.x[i, j]) / dx
            dVy = (V.y[i, j+1] - V.y[i, j]) / dy
            divV[i, j] = dVx + dVy
        end
    end
    return nothing
end


function compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
    nx, ny = size(P)
    nx, ny = size(P)

    ### Dirichlet boundary conditions
    # wall normal velocities are zero
    for j = 1:ny
        V.x[1  , j] = 0.
        V.x[end, j] = 0.
    end

    for i = 1:nx
        V.y[i, 1  ] = 0.
        V.y[i, end] = 0.
    end

    ### pressure update
    for j = 1:ny
        for i = 1:nx
            P[i, j] = P₀[i, j] - γ * ((V.x[i+1, j] - V.x[i, j]) / dx + (V.y[i, j+1] - V.y[i, j]) / dy)
        end
    end

    ### residual at cell interfaces
    ## in horizontal (x) direction
    ## including Neumann BC on Vx at top and bottom boundary
    for j = 1:ny  # all values in y direction
        for i = 2:nx  # inner values in x direction
            # stress at horizontally adjacent cell centers
            τxx_r = 2 * η[i  , j] * (V.x[i+1, j] - V.x[i  , j]) / dx
            τxx_l = 2 * η[i-1, j] * (V.x[i  , j] - V.x[i-1, j]) / dx

            # stress at vertically adjacent cell corners
            if j > 1
                η_b   = 0.25 * (η[i-1, j-1] + η[i, j-1] + η[i-1, j] + η[i, j])
                τxy_b = η_b * ((V.x[i, j] - V.x[i  , j-1]) / dy
                             + (V.y[i, j] - V.y[i-1, j  ]) / dx)
            else
                τxy_b = 0.  # zero stress at the bottom boundary
            end

            if j < ny
                η_t   = 0.25 * (η[i-1, j] + η[i, j] + η[i-1, j+1] + η[i, j+1])
                τxy_t = η_t * ((V.x[i, j+1] - V.x[i  , j  ]) / dy
                             + (V.y[i, j+1] - V.y[i-1, j+1]) / dx)
            else
                τxy_t = 0.  # zero stress at the top boundary
            end


            # residual in x direction on the interface
            R.x[i, j]  = ( (τxx_r - τxx_l) / dx
                         + (τxy_t - τxy_b) / dy
                         - (P[i, j] - P[i-1, j]) / dx)
        end
    end

    ## in vertical (y) direction
    ## including Neumann BC on Vy at left and right boundary
    for j = 2:ny  # inner values in y direction
        for i = 1:nx  # all values in x direction
            τyy_t = 2 * η[i, j  ] * (V.y[i, j+1] - V.y[i, j  ]) / dy
            τyy_b = 2 * η[i, j-1] * (V.y[i, j  ] - V.y[i, j-1]) / dy

            if i > 1
                η_l   = 0.25 * (η[i-1, j-1] + η[i, j-1] + η[i-1, j] + η[i, j])
                τxy_l = η_l * ((V.x[i, j] - V.x[i  , j-1]) / dy
                             + (V.y[i, j] - V.y[i-1, j  ]) / dx)
            else
                τxy_l = 0.  # zero stress at the left boundary
            end

            if i < nx
                η_r   = 0.25 * (η[i, j-1] + η[i+1, j-1] + η[i, j] + η[i+1, j])
                τxy_r = η_r * ((V.x[i+1, j] - V.x[i+1, j-1]) / dy
                             + (V.y[i+1, j] - V.y[i  , j  ]) / dx)
            else
                τxy_r = 0.  # zero stress at the right boundary
            end
            
            R.y[i, j] = ( (τyy_t - τyy_b) / dy
                        + (τxy_r - τxy_l) / dx
                        - ( P[i, j] -  P[i, j-1]) / dy
                        - (ρg[i, j] + ρg[i, j-1]) * 0.5)
        end
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
    return nothing
end


function update_D!(D, R, invM, β)
    for j = 1:size(D.x, 2)
        for i = 2:size(D.x, 1)-1
            D.x[i, j] = invM.x[i, j] * R.x[i, j] + β * D.x[i, j]
        end
    end

    for j = 2:size(D.y, 2)-1
        for i = 1:size(D.y, 1)
            D.y[i, j] = invM.y[i, j] * R.y[i, j] + β * D.y[i, j]
        end
    end
    return nothing
end


function compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Q
    V̄.x .= D.x  # need to copy D since autodiff may change it
    V̄.y .= D.y
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
             Duplicated(P, P̄), Const(P₀), Duplicated(V, V̄),
             Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # -> since R = rhs - A*V, ∂R/∂V * D = -A * D
    #    therefore we use here the negative of the Jacobian-vector product
    return  μ / (dot(D.x, -Q.x) + dot(D.y, -Q.y))
end


function update_V!(V, D, α)
    for j = 1:size(V.x, 2)
        for i = 2:size(V.x, 1)-1
            V.x[i, j] += α * D.x[i, j]
        end
    end
    for j = 2:size(V.y, 2)-1
        for i = 1:size(V.y, 1)
            V.y[i, j] += α * D.y[i, j]
        end
    end
    return nothing
end


function initialise_invM(invM, η, dx, dy, γ)
    nx, ny = size(η)

    for j = 2:ny-1
        for i = 2:nx
            mij = ((2 / dx^2 + 1 / 2dy^2) * (η[i-1, j] + η[i, j])
                  + 1 / 4dy^2 * (η[i-1, j-1] + η[i-1, j+1] + η[i, j-1] + η[i, j+1])
                  + 2 * γ / dx^2)
            invM.x[i, j] = inv(mij)
        end
    end
    # y direction
    for j = 2:ny
        for i = 2:nx-1
            mij = ((2 / dy^2 + 1 / 2dx^2) * (η[i, j-1] + η[i, j])
                  + 1 / 4dx^2 * (η[i-1, j-1] + η[i+1, j-1] + η[i-1, j] + η[i+1, j])
                  + 2 * γ / dy^2)
            invM.y[i, j] = inv(mij)
        end
    end

    ## Neumann boundary points
    # x direction
    for i = 2:nx
        invM.x[i, 1 ] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, 1] + η[i, 1])
                            + 1 / 4dy^2 * (η[i-1, 2] + η[i, 2])
                            + 2 * γ / dx^2)
        invM.x[i, ny] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, ny] + η[i, ny])
                            + 1 / 4dy^2 * (η[i-1, ny-1] + η[i, ny-1])
                            + 2 * γ / dx^2)
    end
    # y direction
    for j = 2:ny
        invM.y[1 , j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[1, j-1] + η[1, j])
                            + 1 / 4dx^2 * (η[2, j-1] + η[2, j])
                            + 2 * γ / dy^2)
        invM.y[nx, j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[nx, j-1] + η[nx, j])
                            + 1 / 4dx^2 * (η[nx-1, j-1] + η[nx-1, j])
                            + 2 * γ / dy^2)
    end

    ## Dirichlet boundary points, leave zero

    return nothing
    
end


function linearStokes2D(; n=127,
                        η_ratio=0.1,
                        niter_in=1000, niter_out=100, ncheck=100,
                        γ_factor=1.,
                        ϵ_in=1e-3,ϵ_max=1e-6, verbose=false)
    η_avg  = 1. # -> Pa s
    ρg_avg = 1. # -> Pa / m
    Lx     = 1. # -> m

    nx = ny = n
    R_in    = 0.1 * Lx
    Ly      = Lx

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)


    # compute viscosities
    A_in  = π * R_in^2
    A_tot = Lx * Ly
    η_out = η_avg * A_tot / (A_tot + (η_ratio - 1)*A_in)
    η_in  = η_out * η_ratio
    # body force
    Δρg   = ρg_avg * A_tot / A_in

    # field initialisation
    η    = [x^2 + y^2 < R_in^2 ? η_in : η_out for x=xs, y=ys]  
    ρg   = [x^2 + y^2 < R_in^2 ? Δρg : 0. for x=xs, y=ys]
    P    = zeros(nx, ny)
    P₀   = zeros(nx, ny)  # old pressure
    P̄    = zeros(nx, ny)  # memory needed for auto-differentiation
    divV = zeros(nx, ny)
    V    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    V̄    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # memory needed for auto-differentiation
    D    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    
    # Coefficient of augmented Lagrangian
    γ = γ_factor * max(η_in, η_out) # η_avg

    # preconditioner
    initialise_invM(invM, η, dx, dy, γ)

    # visualisation
    res_out    = []
    res_in     = []
    conv_in    = []
    itercounts = []
    cg_count   = [1]

    # residual norms for monitoring convergence
    r_out = Inf
    r_in  = Inf
    dP    = Inf
    normP = 0.
    δ_ref = norm(ρg, Inf)

    # outer loop, Powell Hestenes
    it_out  = 1
    while it_out <= niter_out && (dP > 1e-10 && dP > ϵ_max * normP)
        verbose && println("Iteration ", it_out)
        P₀ .= P

        # inner loop, Conjugate Gradient
        
        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)

        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        # δ = sqrt(μ) / δ_ref
        δ = tplNorm(R, Inf) / δ_ref
        # δ = tplNorm(D, Inf) / tplNorm(R, Inf)
        push!(conv_in, δ)
        if it_out > 1 push!(cg_count, cg_count[end]) end
        # start iteration
        it_in = 1
        while it_in <= niter_in
            α = compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
            μ_new = tplDot(R, R, invM)
            β = μ_new / μ
            μ = μ_new
            update_D!(D, R, invM, β)

            # check convergence
            # δ = α * tplNorm(D, Inf) / tplNorm(R, Inf)
            # δ = sqrt(μ) / δ_ref
            δ = tplNorm(R, Inf) / δ_ref
            if δ < min(ϵ_in, max(dP / normP, ϵ_max))
                push!(conv_in, δ)
                push!(cg_count, cg_count[end] + (it_in % ncheck))
                it_in += 1
                break
            end
            if it_in % ncheck == 0
                push!(conv_in, δ)
                push!(cg_count, cg_count[end] + ncheck)
            end
            it_in += 1
        end
        it_in -= 1
        push!(itercounts, it_in)
        verbose && println("finished after ", it_in, " iterations: ")
        it_in == niter_in && println("CG did not reach prescribed accuracy (", δ," > ", min(ϵ_in, max(dP / normP, ϵ_max)) , ")")

        # correction based termination
        dP = norm(P - P₀, Inf)
        normP = norm(P, Inf)

        # record residuals
        r_in = tplNorm(R, Inf) / tplNorm(V, Inf)
        push!(res_in, r_in)
        compute_divV!(divV, V, dx, dy)
        r_out = norm(divV, Inf) / normP
        push!(res_out, r_out)
       

        if verbose
            println("ΔP = ", dP, ", ΔP / |P| = ", dP / normP)
            println("δ  = ", δ)
            println("|Rₚ| / |P| = ", r_in, ", |Rᵥ| / |V| = ", r_in)

            ph_count = cumsum(itercounts)

            fig = Figure(size=(800, 900))
            colours = resample(ColorSchemes.viridis,7)[[2, 4, 6]]
            axs = (P=Axis(fig[1,1][1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
                    Rp=Axis(fig[1,1][1,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure Residual"),
                    Vy=Axis(fig[2,1][1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
                    Ry=Axis(fig[2,1][1,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual"),
                    conv=Axis(fig[3,1], xlabel="iter / nx", ylabel="log error", title="Convergence behaviour"))
            plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:PRGn, colorrange=(-maximum(P), maximum(P))),
                    Rp=image!(axs.Rp, (xs[1], xs[end]), (ys[1], ys[end]), divV,  colormap=:viridis),
                    Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:PRGn, colorrange=(-maximum(V.y), maximum(V.y))),
                    Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), R.y, colormap=:viridis),
                    conv=lines!(axs.conv, cg_count ./ nx, log10.(conv_in), color=colours[3], label="max(Rv)/max(ρg)"))
            scatter!(axs.conv, ph_count ./ nx, log10.(res_out), color=colours[1], marker=:circle, label="Pressure")
            scatter!(axs.conv, ph_count ./ nx, log10.(res_in), color=colours[2], marker=:diamond, label="Velocity")
            
        Colorbar(fig[1, 1][1, 1][1, 2], plt.P)
        Colorbar(fig[1, 1][1, 2][1, 2], plt.Rp)
        Colorbar(fig[2, 1][1, 1][1, 2], plt.Vy)
        Colorbar(fig[2, 1][1, 2][1, 2], plt.Ry)
        axislegend(axs.conv, position=:lb)
        display(fig)
        end
        it_out += 1
    end

    return P, V, R, res_in, res_out, conv_in, itercounts, xs, ys
end


function create_output_plot(P, V, R, errs_in, errs_out, conv_cg, itercounts, xs, ys; ncheck, η_ratio, gamma, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Relative Residuals (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    # compute location of cg iteration errors
    iters_cg  = []
    cg_tot = 0
    for it_tot = iters_out
        while cg_tot < it_tot
            push!(iters_cg, cg_tot)
            cg_tot += ncheck
        end
        cg_tot = it_tot
        push!(iters_cg, cg_tot)
    end

    colours = resample(ColorSchemes.viridis,7)[[2, 4, 6]]
    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=colours[1], marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=colours[2], marker=:diamond, label="Velocity")
    # color limits
    pmax = maximum(abs.(P))
    vmax = maximum(abs.(V.y))
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:PRGn, colorrange=(-pmax, pmax)),
           err=lines!(axs.err, iters_cg ./ nx, log10.(conv_cg), color=colours[3], label="CG conv."),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:PRGn, colorrange=(-vmax, vmax)),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:viridis))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    Legend(fig[1, 2][2, 1], axs.err, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))

    if savefig
        save("2_output_$(η_ratio)_$(gamma).png", fig)
    else
        display(fig)
    end
    return nothing
end


ratio = 0.1
n     = 127
ninner=2*n*n
nouter=10*n
ncheck=n
gamma =10.

outfields = linearStokes2D(n=n,
                           η_ratio=ratio,
                           niter_in=ninner, niter_out=nouter, ncheck=ncheck,
                           γ_factor=gamma,
                           ϵ_in=1e-3,
                           ϵ_max=1e-6, 
                           verbose=false);

create_output_plot(outfields...; ncheck=ncheck, η_ratio=ratio, gamma=gamma, savefig=true)
