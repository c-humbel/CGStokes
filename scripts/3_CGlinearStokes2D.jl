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


function compute_R!(R, P, V, ρg, η, dx, dy)
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

    ### pressure residual
    for j = 1:ny
        for i = 1:nx
            dVx = (V.x[i+1, j] - V.x[i, j]) / dx
            dVy = (V.y[i, j+1] - V.y[i, j]) / dy
            R.p[i, j] = dVx + dVy
        end
    end

    ### velocity residual at cell interfaces
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

    for I = eachindex(D.p)
        D.p[I] = R.p[I] + β * D.p[I]
    end
    return nothing
end


function compute_α(R, Q, P, P̄, V, V̄, D, ρg, η, μ, dx, dy,)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Q
    V̄.x .= D.x  # need to copy D since autodiff may change it
    V̄.y .= D.y
    P̄ .= D.p
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
             Duplicated(P, P̄), Duplicated(V, V̄),
             Const(ρg), Const(η), Const(dx), Const(dy))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # -> since R = rhs - A*V, ∂R/∂V * D = -A * D
    #    therefore we use here the negative of the Jacobian-vector product
    return  μ / tplDot(D, Q, -1)
end


function update_P_V!(P, V, D, α)
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

    for I = eachindex(P)
        P[I] += α * D.p[I]
    end
    return nothing
end


function initialise_invM(invM, η, dx, dy)
    nx, ny = size(η)

    for j = 2:ny-1
        for i = 2:nx
            mij = ((2 / dx^2 + 1 / 2dy^2) * (η[i-1, j] + η[i, j])
                  + 1 / 4dy^2 * (η[i-1, j-1] + η[i-1, j+1] + η[i, j-1] + η[i, j+1]))
            invM.x[i, j] = inv(mij)
        end
    end
    # y direction
    for j = 2:ny
        for i = 2:nx-1
            mij = ((2 / dy^2 + 1 / 2dx^2) * (η[i, j-1] + η[i, j])
                  + 1 / 4dx^2 * (η[i-1, j-1] + η[i+1, j-1] + η[i-1, j] + η[i+1, j]))
            invM.y[i, j] = inv(mij)
        end
    end

    ## Neumann boundary points
    # x direction
    for i = 2:nx
        invM.x[i, 1 ] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, 1] + η[i, 1])
                            + 1 / 4dy^2 * (η[i-1, 2] + η[i, 2]))
        invM.x[i, ny] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, ny] + η[i, ny])
                            + 1 / 4dy^2 * (η[i-1, ny-1] + η[i, ny-1]))
    end
    # y direction
    for j = 2:ny
        invM.y[1 , j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[1, j-1] + η[1, j])
                            + 1 / 4dx^2 * (η[2, j-1] + η[2, j]))
        invM.y[nx, j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[nx, j-1] + η[nx, j])
                            + 1 / 4dx^2 * (η[nx-1, j-1] + η[nx-1, j]))
    end

    ## Dirichlet boundary points, leave zero

    return nothing
end


function linearStokes2D(; n=127,
                        η_in=0.1, η_out=1., ρg_in=1.,
                        maxiter=10000, ncheck=100,
                        ϵ_max=1e-6)
    L_ref = 10. # reference length 
    η_ref = max(η_in, η_out)
    ρg_mag = min(1., η_out / η_in)
    ρg_ref = ρg_in / ρg_mag
    Lx = Ly = 1.
    R_in  = 0.1
    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # field initialisation
    η    = [x^2 + y^2 < R_in^2 ? η_in / η_ref : η_out / η_ref for x=xs, y=ys]  
    ρg   = [x^2 + y^2 < R_in^2 ? ρg_mag : 0.    for x=xs, y=ys]
    P    = zeros(nx, ny)
    P̄    = zeros(nx, ny)  # memory needed for auto-differentiation
    V    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    V̄    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # memory needed for auto-differentiation
    D    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1), p=zeros(nx, ny))  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1), p=zeros(nx, ny))  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1), p=zeros(nx, ny))  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = (x=zeros(nx+1, ny), y=zeros(nx, ny+1), p=ones(nx, ny))   # preconditioner

    # visualisation
    residuals = []
    itercount = 0

    # initialise preconditioner
    initialise_invM(invM, η, dx, dy)

    # iteration zero
    compute_R!(R, P, V, ρg, η, dx, dy)

    tplSet!(D, R, invM)
    μ = tplDot(R, D)
    # start iteration
    for it = 1:maxiter
        α = compute_α(R, Q, P, P̄, V, V̄, D, ρg, η, μ, dx, dy)
        update_P_V!(P, V, D, α)
        compute_R!(R, P, V, ρg, η, dx, dy)
        μ_new = tplDot(R, R, invM)
        β = μ_new / μ
        μ = μ_new
        update_D!(D, R, invM, β)

        # check convergence
        if μ < ϵ_max^2
            itercount = it
            push!(residuals, sqrt(μ))
            break
        end
        if it % ncheck == 0
            println("iteration ", it, "; residual = ", sqrt(μ))
            push!(residuals, sqrt(μ))
        end
    end
    if itercount == 0 itercount = maxiter end
    println("finished after ", itercount, " iterations with L2-residual: ", residuals[end]) 

    # scale output variables
    P.*= ρg_ref * L_ref
    tplScale!(V, ρg_ref * L_ref^2 / η_ref)
    tplScale!(R, ρg_ref)

    return P, V, R, residuals, itercount, xs .* L_ref, ys .* L_ref
end


function create_output_plot(P, V, R, residuals, itercount, xs, ys; ncheck, η_ratio, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="CG Convergence (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of cg iteration errors
    iters_cg  = [i for i in ncheck:ncheck:itercount]
    if itercount % ncheck != 0
        push!(iters_cg, itercount)
    end

    # color limits
    pmax = maximum(abs.(P))
    vmax = maximum(abs.(V.y))
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:PRGn, colorrange=(-pmax, pmax)),
           err=lines!(axs.err, iters_cg ./ nx, log10.(residuals), color=:green),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:PRGn, colorrange=(-vmax, vmax)),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:viridis))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)

    if savefig
        save("3_output_$(η_ratio).png", fig)
    else
        display(fig)
    end
    return nothing
end



eta_outer = 1.
eta_inner = 0.1
n     = 127
niter = 100000
ncheck=100

outfields = linearStokes2D(n=n,
                           η_in=eta_inner, η_out=eta_outer, ρg_in=-1.,
                           maxiter=niter, ncheck=ncheck,
                           ϵ_max=1e-6);

create_output_plot(outfields...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, savefig=true)

