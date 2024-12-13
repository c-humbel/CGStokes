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
                         - (ρg.c[i, j] + ρg.c[i, j-1]) * 0.5)
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


function linearStokes2D(;n=127,
                        η_in=0.1, η_out=1., ρg_in=1.,
                        niter_in=10000, niter_out=100, ncheck=100,
                        γ_factor=1.,
                        ϵ_in=1e-3, ϵ_max=1e-6, verbose=false)
    L_ref = 10. # reference length 
    η_ref = max(η_in, η_out)
    ρg_mag = min(1., η_out / η_in)
    ρg_ref = ρg_in / ρg_mag
    Lx = Ly = 1.
    R_in  = 0.1
    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    xc = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    yc = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)

    # field initialisation
    η    = (c=[x^2 + y^2 < R_in^2 ? η_in / η_ref : η_out / η_ref for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? η_in / η_ref : η_out / η_ref for x=xv, y=yv]) 
    ρg   = (c=[x^2 + y^2 < R_in^2 ? ρg_mag : 0. for x=xc, y=yc],
            v=[x^2 + y^2 < R_in^2 ? ρg_mag : 0. for x=xv, y=yv])
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)
    V    = (xc=zeros(nx+1, ny), yc=zeros(nx, ny+1), xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by search vector D
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    
    # Coefficient of augmented Lagrangian
    γ = γ_factor # maximum(η), always one

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
    δ     = Inf

    # outer loop, Powell Hestenes
    it_out  = 1
    while it_out <= niter_out && (r_out > ϵ_max || r_in > ϵ_max)
        verbose && println("Iteration ", it_out)
        tplSet!(P₀, P)

        # inner loop, Conjugate Gradient
        
        # iteration zero
        compute_R!(R, P, P₀, V, ρg, η, dx, dy, γ)

        tplSet!(D, R, invM)
        μ = tplDot(R, D)
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
            δ = α * tplNorm(D, Inf) / tplNorm(R, Inf)
            if δ < min(ϵ_in, r_out)
                it_in += 1
                push!(conv_in, δ)
                break
            end
            if it_in % ncheck == 0
                push!(conv_in, δ)
            end
            it_in += 1
        end
        it_in -= 1
        push!(itercounts, it_in)
        verbose && println("finished after ", it_in, " iterations: ")

        r_in = tplNorm(R, Inf)
        push!(res_in, r_in)
    
        compute_divV!(divV, V, dx, dy)
        r_out = tplNorm(divV, Inf)
        push!(res_out, r_out)

        verbose && println("p-residual = ", r_out)
        verbose && println("v-residual = ", r_in)
        it_out += 1
    end

    # scale output variables
    tplScale!(P, ρg_ref * L_ref)
    tplScale!(V, ρg_ref * L_ref^2 / η_ref)
    tplScale!(R, ρg_ref)

    return P, V, R, res_in, res_out, conv_in, itercounts, xc .* L_ref, yc .* L_ref
end


function create_output_plot(P, V, R, errs_in, errs_out, conv_cg, itercounts, xs, ys; ncheck, η_ratio, gamma, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P.c, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Residual Norm (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    # compute location of cg iteration errors
    iters_cg  = []
    cg_tot = ncheck
    for it_tot = iters_out
        while cg_tot < it_tot
            push!(iters_cg, cg_tot)
            cg_tot += ncheck
        end
        cg_tot = it_tot
    end
    push!(iters_cg, cg_tot)

    colours = resample(ColorSchemes.viridis,7)[[2, 4, 6]]
    pmax = maximum(abs.(P.c))
    vmax = maximum(abs.(V.yc))

    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=colours[1], marker=:circle, label="Pressure")
    scatter!(axs.err, iters_out ./ nx, log10.(errs_in), color=colours[2], marker=:diamond, label="Velocity")
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P.c, colormap=:PRGn, colorrange=(-pmax, pmax)),
           err=lines!(axs.err, iters_cg ./ nx, log10.(conv_cg), color=colours[3], label="CG"),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.yc, colormap=:PRGn, colorrange=(-vmax, vmax)),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.yc)), colormap=:viridis))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    Legend(fig[1, 2][2, 1], axs.err, orientation=:horizontal, framevisible=false, padding=(0, 0, 0, 0))

    if savefig
        save("2b_result_$(η_ratio)_$(gamma).png", fig)
    else
        display(fig)
    end
    return nothing
end


eta_outer = 1.
eta_inner = 0.1
n     = 127
ninner=10000
nouter=100
ncheck=100
gamma =20.

outfields = linearStokes2D(n=n,
                           η_in=eta_inner, η_out=eta_outer, ρg_in=-1.,
                           niter_in=ninner, niter_out=nouter, ncheck=ncheck,
                           γ_factor=gamma,
                           ϵ_in=1e-7,
                           ϵ_max=1e-6, 
                           verbose=true);

create_output_plot(outfields...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, gamma=gamma, savefig=true)
