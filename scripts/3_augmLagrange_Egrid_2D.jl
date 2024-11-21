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
    nx, ny = size(divV.c)
    for j = 1:ny
        for i = 1:nx
            dVx = (V.xc[i+1, j] - V.xc[i, j]) / dx
            dVy = (V.yc[i, j+1] - V.yc[i, j]) / dy
            divV.c[i, j] = dVx + dVy
        end
    end

    for j = 1:ny+1
        for i = 1:nx+1
            dVx = (V.xv[i+1, j] - V.xv[i, j]) / dx
            dVy = (V.yv[i, j+1] - V.yv[i, j]) / dy
            divV.v[i, j] = dVx + dVy
        end
    end

    return nothing
end


function compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)
    nx, ny = size(P.c)

    ### Dirichlet boundary conditions
    # wall normal velocities are zero
    for j = 1:ny
        V.xc[1  , j] = 0.
        V.xc[end, j] = 0.
    end

    for j = 1:ny+1
        V.xv[1  , j] = 0.
        V.xv[end, j] = 0.
    end

    for i = 1:nx
        V.yc[i, 1  ] = 0.
        V.yc[i, end] = 0.
    end

    for i = 1:nx+1
        V.yv[i, 1  ] = 0.
        V.yv[i, end] = 0.
    end

    ### pressure update
    for j = 1:ny
        for i = 1:nx
            P.c[i, j] = P₀.c[i, j] - γ * ((V.xc[i+1, j] - V.xc[i, j]) / dx + (V.yc[i, j+1] - V.yc[i, j]) / dy)
        end
    end

    for j = 1:ny+1
        for i = 1:nx+1
            P.v[i, j] = P₀.v[i, j] - γ * ((V.xv[i+1, j] - V.xv[i, j]) / dx + (V.yv[i, j+1] - V.yv[i, j]) / dy)
        end
    end

    ### residual in horizontal (x) directio
    ## including Neumann BC on at top and bottom boundary
    ## for velocities associated with cell centers (V.xc)
    for j = 1:ny  # all values in y direction
        for i = 2:nx  # inner values in x direction
            # stress at horizontally adjacent cell centers
            τxx_r = 2 * η.c[i  , j] * (V.xc[i+1, j] - V.xc[i  , j]) / dx
            τxx_l = 2 * η.c[i-1, j] * (V.xc[i  , j] - V.xc[i-1, j]) / dx

            # stress at vertically adjacent cell corners
            if j > 1
                τxy_b = η.v[i, j  ] * ((V.xc[i, j  ] - V.xc[i, j-1]) / dy + (V.yc[i, j  ] - V.yc[i-1, j  ]) / dx)
            else
                τxy_b = 0.  # zero stress at the bottom boundary
            end

            if j < ny
                τxy_t = η.v[i, j+1] * ((V.xc[i, j+1] - V.xc[i, j  ]) / dy + (V.yc[i, j+1] - V.yc[i-1, j+1]) / dx)
            else
                τxy_t = 0.  # zero stress at the top boundary
            end


            # residual in x direction on the interface
            R.xc[i, j]  = ( (τxx_r - τxx_l) / dx
                          + (τxy_t - τxy_b) / dy
                          - (P.c[i, j] - P.c[i-1, j]) / dx)
        end
    end
    ## for velocities associated with cell corners (V.xv)
    for j = 1:ny+1  # all values in y direction
        for i = 2:nx+1  # inner values in x direction
            τxx_r = 2 * η.v[i  , j] * (V.xv[i+1, j] - V.xv[i  , j]) / dx
            τxx_l = 2 * η.v[i-1, j] * (V.xv[i  , j] - V.xv[i-1, j]) / dx

            if j > 1
                τxy_b = η.c[i-1, j-1] * ((V.xv[i, j  ] - V.xv[i, j-1]) / dy + (V.yv[i, j  ] - V.yv[i-1, j  ]) / dx)
            else
                τxy_b = 0.  # zero stress at the bottom boundary
            end

            if j < ny+1
                τxy_t = η.c[i-1, j  ] * ((V.xv[i, j+1] - V.xv[i, j  ]) / dy + (V.yv[i, j+1] - V.yv[i-1, j+1]) / dx)
            else
                τxy_t = 0.  # zero stress at the top boundary
            end

            R.xv[i, j] = ( (τxx_r - τxx_l) / dx
                         + (τxy_t - τxy_b) / dy
                         - (P.v[i, j] - P.v[i-1, j]) / dx)
        end
    end

    ### residual in vertical (y) direction
    ### including Neumann BC at left and right boundary
    ## for velocities associated with cell centers (V.yc)
    for j = 2:ny  # inner values in y direction
        for i = 1:nx  # all values in x direction
            τyy_t = 2 * η.c[i, j  ] * (V.yc[i, j+1] - V.yc[i, j  ]) / dy
            τyy_b = 2 * η.c[i, j-1] * (V.yc[i, j  ] - V.yc[i, j-1]) / dy

            if i > 1
                τxy_l = η.v[i  , j] * ((V.xc[i  , j] - V.xc[i  , j-1]) / dy + (V.yc[i  , j] - V.yc[i-1, j]) / dx)
            else
                τxy_l = 0.  # zero stress at the left boundary
            end

            if i < nx
                τxy_r = η.v[i+1, j] * ((V.xc[i+1, j] - V.xc[i+1, j-1]) / dy + (V.yc[i+1, j] - V.yc[i  , j]) / dx)
            else
                τxy_r = 0.  # zero stress at the right boundary
            end
            
            R.yc[i, j] = ( (τyy_t - τyy_b) / dy
                         + (τxy_r - τxy_l) / dx
                         - ( P.c[i, j] -  P.c[i, j-1]) / dy
                         + (ρg.c[i, j] + ρg.c[i, j-1]) * 0.5)
        end
    end
    ## for velocities associated with cell corners (V.yv)
    for j=2:ny+1
        for i=1:nx+1
            τyy_t = 2 * η.v[i, j  ] * (V.yv[i, j+1] - V.yv[i, j  ]) / dy
            τyy_b = 2 * η.v[i, j-1] * (V.yv[i, j  ] - V.yv[i, j-1]) / dy

            if i > 1
                τxy_l = η.c[i-1, j-1] * ((V.xv[i  , j] - V.xv[i  , j-1]) / dy + (V.yv[i  , j] - V.yv[i-1, j]) / dx)
            else
                τxy_l = 0.  # zero stress at the left boundary
            end

            if i < nx+1
                τxy_r = η.c[i  , j-1] * ((V.xv[i+1, j] - V.xv[i+1, j-1]) / dy + (V.yv[i+1, j] - V.yv[i  , j]) / dx)
            else
                τxy_r = 0.  # zero stress at the right boundary
            end

            R.yv[i, j] = ( (τyy_t - τyy_b) / dy
                         + (τxy_r - τxy_l) / dx
                         - ( P.v[i, j] -  P.v[i, j-1]) / dy
                         + (ρg.v[i, j] + ρg.v[i, j-1]) * 0.5)
        end
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
    return nothing
end


function update_D!(D, R, invM, β)
    for j = 1:size(D.xc, 2)
        for i = 2:size(D.xc, 1)-1
            D.xc[i, j] = invM.xc[i, j] * R.xc[i, j] + β * D.xc[i, j]
        end
    end

    for j = 2:size(D.yc, 2)-1
        for i = 1:size(D.yc, 1)
            D.yc[i, j] = invM.yc[i, j] * R.yc[i, j] + β * D.yc[i, j]
        end
    end

    for j = 1:size(D.xv, 2)
        for i = 2:size(D.xv, 1)-1
            D.xv[i, j] = invM.xv[i, j] * R.xv[i, j] + β * D.xv[i, j]
        end
    end

    for j = 2:size(D.yv, 2)-1
        for i = 1:size(D.yv, 1)
            D.yv[i, j] = invM.yv[i, j] * R.yv[i, j] + β * D.yv[i, j]
        end
    end
    return nothing
end


function compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Q
    tplSet!(V̄, D) # need to copy D since autodiff may change it
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
             Duplicated(P, P̄), Const(P₀), Duplicated(V, V̄),
             Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # -> since R = rhs - A*V, ∂R/∂V * D = -A * D
    #    therefore we use here the negative of the Jacobian-vector product
    return  μ / tplDot(D, Q, -1.)
end


function update_V!(V, D, α)
    for j = 1:size(V.xc, 2)
        for i = 2:size(V.xc, 1)-1
            V.xc[i, j] += α * D.xc[i, j]
        end
    end
    for j = 2:size(V.yc, 2)-1
        for i = 1:size(V.yc, 1)
            V.yc[i, j] += α * D.yc[i, j]
        end
    end

    for j = 1:size(V.xv, 2)
        for i = 2:size(V.xv, 1)-1
            V.xv[i, j] += α * D.xv[i, j]
        end
    end

    for j = 2:size(V.yv, 2)-1
        for i = 1:size(V.yv, 1)
            V.yv[i, j] += α * D.yv[i, j]
        end
    end
    return nothing
end


function initialise_invM(invM, η, dx, dy, γ)
    nx, ny = size(η.c)

    ## inner points
    # x direction, cell centers
    for j = 2:ny-1
        for i = 2:nx
            mij = ( 2 / dx^2 * (η.c[i-1, j] + η.c[i, j  ])
                  + 1 / dy^2 * (η.v[i  , j] + η.v[i, j+1])
                  + 2 * γ / dx^2)
            invM.xc[i, j] = inv(mij)
        end
    end

    # y direction, cell centers
    for j = 2:ny
        for i = 2:nx-1
            mij = ( 2 / dy^2 * (η.c[i, j-1] + η.c[i  , j])
                  + 1 / dx^2 * (η.v[i, j  ] + η.v[i+1, j])
                  + 2 * γ / dy^2)
            invM.yc[i, j] = inv(mij)
        end
    end

    # x direction, vertices
    for j = 2:ny
        for i = 2:nx+1
            mij = ( 2 / dx^2 * (η.v[i-1, j  ] + η.v[i  , j])
                  + 1 / dy^2 * (η.c[i-1, j-1] + η.c[i-1, j])
                  + 2 * γ / dx^2)
            invM.xv[i, j] = inv(mij)
        end
    end

    # y direction, vertices
    for j=2:ny+1
        for i=2:nx
            mij = ( 2 / dy^2 * (η.v[i  , j-1] + η.v[i, j  ])
                  + 1 / dx^2 * (η.c[i-1, j-1] + η.c[i, j-1])
                  + 2 * γ / dy^2)
            invM.yv[i, j] = inv(mij)
        end
    end

    ## Neumann boundary points
    # x direction, cell centers
    for i = 2:nx
        invM.xc[i, 1 ] = inv( 2 / dx^2 * (η.c[i-1, 1] + η.c[i, 1])
                            + 1 / dy^2 * (η.v[i, 2])
                            + 2 * γ / dx^2)
        invM.xc[i, ny] = inv( 2 / dx^2 * (η.c[i-1, ny] + η.c[i, ny])
                            + 1 / dy^2 * η.v[i, ny]
                            + 2 * γ / dx^2)
    end
    # y direction, cell centers
    for j = 2:ny
        invM.yc[1 , j] = inv( 2 / dy^2 * (η.c[1, j-1] + η.c[1, j])
                            + 1 / dx^2 * (η.v[2, j])
                            + 2 * γ / dy^2)
        invM.yc[nx, j] = inv( 2 / dy^2  * (η.c[nx, j-1] + η.c[nx, j])
                            + 1 / dx^2 * (η.v[nx, j])
                            + 2 * γ / dy^2)
    end
    # x direction, vertices
    for i = 2:nx+1
        invM.xv[i, 1   ] = inv( 2 / dx^2 * (η.v[i-1, 1] + η.v[i, 1])
                              + 1 / dy^2 * (η.c[i-1, 1])
                              + 2 * γ / dx^2)
        invM.xv[i, ny+1] = inv( 2 / dx^2 * (η.v[i-1, ny+1] + η.v[i, ny+1])
                              + 1 / dy^2 * (η.c[i-1, ny  ])
                              + 2 * γ / dx^2)
    end
    # y direction, vertices
    for j = 2:ny+1
        invM.yv[1   , j] = inv( 2 / dy^2 * (η.v[1, j-1] + η.v[1, j])
                              + 1 / dx^2 * (η.c[1, j-1])
                              + 2 * γ / dy^2)
        invM.yv[nx+1, j] = inv( 2 / dy^2 * (η.v[nx+1, j-1] + η.v[nx+1, j])
                              + 1 / dx^2 * (η.c[nx  , j-1])
                              + 2 * γ / dy^2)
    end

    ## Dirichlet boundary points, leave zero

    return nothing
    
end


function linearStokes2D(; n=127,
                        η_in=0.1, η_out=1., ρg_in=1.,
                        niter_in=1000, niter_out=100, ncheck=100,
                        γ_factor=1.,
                        ϵ_in=1e-3,ϵ_max=1e-5)
                        
    Lx = Ly = 10.
    R_in  = 1.
    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    # coordinates of cell centers
    xc = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    yc = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)
    # coordinates of cell corners (vertices)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)

    # field initialisation
    η    = (c=[x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xv, y=yv])
    ρg   = (c=[x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xv, y=yv])
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)  # divergence of velocity
    V    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    
    # Coefficient of augmented Lagrangian
    γ = γ_factor*max(maximum(η.c), maximum(η.v))

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
        tplSet!(P₀, P)

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
        r_out = tplNorm(divV, Inf)
        push!(res_out, r_out)
        println("p-residual = ", r_out)
        println("v-residual = ", r_in)
        it_out += 1
    end

    return P, V, R, res_in, res_out, conv_in, itercounts, xc, yc
end


# TODO: update for Arakawa grid
function create_output_plot(P, V, R, errs_in, errs_out, conv_cg, itercounts, xs, ys; ncheck, η_ratio, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P.c, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Residual Norm (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    iters_cg  = ncheck .* (1:length(conv_cg))
    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=:blue, marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=:green, marker=:diamond, label="Velocity")
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P.c, colormap=:inferno),
           err=lines!(axs.err, iters_cg ./ nx, log10.(conv_cg), color=:red, label="CG"),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.yc, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.yc)), colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    axislegend(axs.err, position=:lb)

    if savefig
        save("3_output_$(η_ratio)_$(maximum(itercounts)).png", fig)
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
    lines!(ax, iters_cg ./ nx, log10.(conv_cg), color=:red, label="CG convergence")
    scatter!(ax, iters_out ./ nx, log10.(errs_out), color=:blue, marker=:circle, label="Pressure")
    scatter!(ax, iters_out ./ nx, log10.(errs_in), color=:green, marker=:diamond, label="Velocity")
    axislegend(ax, position=:rt)
    if savefig
        save("3_convergence_$(η_ratio)_$(maximum(itercounts)).png", fig)
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
ncheck=100

outfields = linearStokes2D(n=n,
                           η_in=eta_inner, η_out=eta_outer, ρg_in=eta_outer,
                           niter_in=ninner, niter_out=nouter, ncheck=ncheck,
                           γ_factor=1., ϵ_in=1e-2, ϵ_max=1e-6);

create_output_plot(outfields...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, savefig=false)

create_convergence_plot(outfields[4:7]...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, nx=n, savefig=false)
