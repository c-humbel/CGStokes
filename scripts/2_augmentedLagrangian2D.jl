using CairoMakie
using LinearAlgebra
using Enzyme


function tplNorm(x::NamedTuple, p::Real=2)
    return norm(norm.(values(x), p), p)   
end


function tplDot(x::NamedTuple, y::NamedTuple, a::Union{NamedTuple, Real}=1.)
    if a isa NamedTuple
        s = 0.
        for k = keys(x)
            s += dot(x[k], a[k] .* y[k])
        end
        return return s
    else
        return sum(dot.(values(x), a .* values(y)))
    end
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::Union{NamedTuple, Real}=1.)
    if a isa NamedTuple
        for k = keys(dest)
            copyto!(dest[k], a[k] .* src[k])
        end
    else
        copyto!.(values(dest), a .* values(src))
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
                        + (ρg[i, j] + ρg[i, j-1]) * 0.5)
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
                        η_in=0.1, η_out=1., ρg_in=1.,
                        niter_in=1000, niter_out=100, ncheck=100,
                        γ_factor=1.,
                        ϵ_in=1e-3,ϵ_max=1e-6)
                        
    Lx = Ly = 10.
    R_in  = 1.
    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # field initialisation
    η    = [x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xs, y=ys]
    ρg   = [x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xs, y=ys]
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
    γ = γ_factor*maximum(η)

    # preconditioner
    initialise_invM(invM, η, dx, dy, γ)

    # visualisation
    res_out    = []
    res_in     = []
    conv_in    = []
    itercounts = []

    # residual norms for monitoring convergence
    r_out = Inf
    r_in  = Inf
    r₀    = Inf
    δ     = Inf

    # outer loop, Powell Hestenes
    it_out  = 1
    while it_out <= niter_out && (r_out > ϵ_max || r_in > ϵ_max)
        println("Iteration ", it_out)
        P₀ .= P

        # inner loop, Conjugate Gradient
        
        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)

        tplSet!(D, R, invM)
        μ = tplDot(R, D)
        r₀ = tplNorm(R, Inf)
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
            
            it_in += 1

            if it_in % ncheck == 0
                δ = α * tplNorm(D, Inf)
                println("\t", it_in, ": δ/r₀ = ", δ / r₀)
                push!(conv_in, δ / r₀)
                if δ < min(ϵ_in, r_out) * r₀ break end
            end
        end
        push!(itercounts, it_in-1)
        
        r_in = tplNorm(R, Inf)
        push!(res_in, r_in)
        compute_divV!(divV, V, dx, dy)
        r_out = norm(divV, Inf)
        push!(res_out, r_out)
        println("p-residual = ", r_out)
        println("v-residual = ", r_in)
        it_out += 1
    end

    return P, V, R, res_in, res_out, conv_in, itercounts, xs, ys
end


function create_output_plot(P, V, R, errs_in, errs_out, conv_cg, itercounts, xs, ys; ncheck, η_ratio, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Residual norm (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    iters_cg  = ncheck .* (1:length(conv_cg))
    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=:blue, marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=:green, marker=:diamond, label="Velocity")
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:inferno),
           err=lines!(axs.err, iters_cg ./ nx, log10.(conv_cg), color=:red, label="CG conv."),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    axislegend(axs.err)

    if savefig
        save("2_output_$(η_ratio)_$(maximum(itercounts)).png", fig)
    else
        display(fig)
    end
    return nothing
end



function create_convergence_plot(errs_in, errs_out, conv_cg, itercounts; ncheck, η_ratio, nx, savefig=false)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Iterations / nx", ylabel="Residual norm (log)", title="η ratio=$η_ratio")
    iters_out = cumsum(itercounts)
    iters_cg  = ncheck .* (1:length(conv_cg))
    lines!(ax, iters_cg ./ nx, log10.(conv_cg), color=:red, label="CG conv.")
    scatter!(ax, iters_out ./ nx, log10.(errs_out), color=:blue, marker=:circle, label="Pressure")
    scatter!(ax, iters_out ./ nx, log10.(errs_in), color=:green, marker=:diamond, label="Velocity")
    axislegend(ax, position=:rt)
    if savefig
        save("2_convergence_$(η_ratio)_$(maximum(itercounts)).png", fig)
    else
        display(fig)
    end
    return nothing
end

eta_outer = 1e-6
eta_inner = 1.
n     = 127
ninner=10000
nouter=100
ncheck=20

outfields = linearStokes2D(n=n,
                           η_in=eta_inner, η_out=eta_outer, ρg_in=eta_outer,
                           niter_in=ninner, niter_out=nouter, ncheck=ncheck,
                           γ_factor=0.1,
                           ϵ_in=1e-3,
                           ϵ_max=1e-6);

create_output_plot(outfields...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, savefig=true)

create_convergence_plot(outfields[4:7]...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, nx=n, savefig=false)
