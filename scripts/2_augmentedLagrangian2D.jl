using CairoMakie
using LinearAlgebra
using Enzyme


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


function compute_R!(R, P, P_old, V, ρg, η, dx, dy, γ)
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
            P[i, j] = P_old[i, j] - γ * ((V.x[i+1, j] - V.x[i, j]) / dx + (V.y[i, j+1] - V.y[i, j]) / dy)
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
            R.x[i, j]  = (- (τxx_r - τxx_l) / dx
                          - (τxy_t - τxy_b) / dy
                          + (P[i, j] - P[i-1, j]) / dx)
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
            
            R.y[i, j] = ( - (τyy_t - τyy_b) / dy
                          - (τxy_r - τxy_l) / dx
                          + ( P[i, j] -  P[i, j-1]) / dy
                          - (ρg[i, j] + ρg[i, j-1]) * 0.5)
        end
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
    return nothing
end


function compute_β_rMr(rMr, R, Minv)
    rMr_new = dot(R.x, Minv.x .* R.x) + dot(R.y, Minv.y .* R.y)
    return rMr_new / rMr, rMr_new
end


function update_D!(D, R, Minv, β)
    for j = 1:size(D.x, 2)
        for i = 2:size(D.x, 1)-1
            D.x[i, j] = Minv.x[i, j] * R.x[i, j] + β * D.x[i, j]
        end
    end

    for j = 2:size(D.y, 2)-1
        for i = 1:size(D.y, 1)
            D.y[i, j] = Minv.y[i, j] * R.y[i, j] + β * D.y[i, j]
        end
    end
    return nothing
end


function compute_α(R, Ad, P, P_tmp, P_old, V, D, ρg, η, rMr, dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Ad
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Ad),
             Duplicated(P, P_tmp), Const(P_old), Duplicated(V, D),
             Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # note that, since R = rhs - A*V, ∂R/∂V * D = -A * D
    # therefore we use here the negative of the Jacobian-vector product
    return  rMr / (dot(D.x, -Ad.x) + dot(D.y, -Ad.y))
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


function initialise_Minv(Minv, η, dx, dy, γ)
    nx, ny = size(η)

    for j = 2:ny-1
        for i = 2:nx
            mij = ((2 / dx^2 + 1 / 2dy^2) * (η[i-1, j] + η[i, j])
                  + 1 / 4dy^2 * (η[i-1, j-1] + η[i-1, j+1] + η[i, j-1] + η[i, j+1])
                  + 2 * γ / dx^2)
            Minv.x[i, j] = inv(mij)
        end
    end
    # y direction
    for j = 2:ny
        for i = 2:nx-1
            mij = ((2 / dy^2 + 1 / 2dx^2) * (η[i, j-1] + η[i, j])
                  + 1 / 4dx^2 * (η[i-1, j-1] + η[i+1, j-1] + η[i-1, j] + η[i+1, j])
                  + 2 * γ / dy^2)
            Minv.y[i, j] = inv(mij)
        end
    end

    ## Neumann boundary points
    # x direction
    for i = 2:nx
        Minv.x[i, 1 ] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, 1] + η[i, 1])
                            + 1 / 4dy^2 * (η[i-1, 2] + η[i, 2])
                            + 2 * γ / dx^2)
        Minv.x[i, ny] = inv((2 / dx^2 + 1 / 4dy^2) * (η[i-1, ny] + η[i, ny])
                            + 1 / 4dy^2 * (η[i-1, ny-1] + η[i, ny-1])
                            + 2 * γ / dx^2)
    end
    # y direction
    for j = 2:ny
        Minv.y[1 , j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[1, j-1] + η[1, j])
                            + 1 / 4dx^2 * (η[2, j-1] + η[2, j])
                            + 2 * γ / dy^2)
        Minv.y[nx, j] = inv((2 / dy^2 + 1 / 4dx^2) * (η[nx, j-1] + η[nx, j])
                            + 1 / 4dx^2 * (η[nx-1, j-1] + η[nx-1, j])
                            + 2 * γ / dy^2)
    end

    ## Dirichlet boundary points, leave zero

    return nothing
    
end

# compute energy norm (r^T M^-1 A r)
# this should be monotonically decreasing in CG iteration
function energy_norm(R_tmp, Ar, P, P_tmp, P_old, V, R, Minv, ρg, η, dx, dy, γ)
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R_tmp, Ar),
             Duplicated(P, P_tmp), Const(P_old), Duplicated(V, R),
             Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    return sqrt(dot(R.x, Ar.x) + dot(R.y, Ar.y))
end




function linearStokes2D(; n=127,
                        η_in=0.1, η_out=1., ρg_in=1.,
                        niter_in=1000, niter_out=100, ncheck=100,
                        γ_factor=1.,
                        max_err=1e-6)
    Lx = Ly = 10.
    R_in  = 1.
    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # field initialisation
    η      = [x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xs, y=ys]
    ρg     = [x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xs, y=ys]
    P      = zeros(nx, ny)
    P_old  = zeros(nx, ny)
    P_tmp  = zeros(nx, ny)
    divV   = zeros(nx, ny)
    V      = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    D      = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # search direction of CG, cells affecting Dirichlet BC are zero
    R      = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Residuals of velocity PDE, cells affected by Dirichlet BC are zero
    R_tmp  = (x=zeros(nx+1, ny), y=zeros(nx, ny+1)) 
    Ad     = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Jacobian of compute_R wrt. V, multiplied by search vector D
    Minv   = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    # Coefficient of augmented Lagrangian
    γ = γ_factor*maximum(η)

    # preconditioner
    initialise_Minv(Minv, η, dx, dy, γ)

    
    # visualisation
    errs_in = []
    errs_out = []
    itercounts = []

    err_out = 2 * max_err
    it_out = 1
    # outer loop, Powell Hestenes
    while it_out <= niter_out && err_out > max_err
        err_in = 2 * max_err
        it_in = 1
        # inner loop, Conjugate Gradient
        # iteration zero
        P_old .= P
        compute_R!(R, P, P_old, V, ρg, η, dx, dy, γ)
        D.x .= Minv.x .* R.x
        D.y .= Minv.y .* R.y
        rMr = dot(R.x, Minv.x .* R.x) + dot(R.y, Minv.y .* R.y)
        while it_in <= niter_in && err_in > max_err
            α = compute_α(R, Ad, P, P_tmp, P_old, V, D, ρg, η, rMr, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, P_old, V, ρg, η, dx, dy, γ)
            β, rMr = compute_β_rMr(rMr, R, Minv)
            update_D!(D, R, Minv, β)

            if it_in % ncheck == 0
                err_in = energy_norm(R_tmp, Ad, P, P_tmp, P_old, V, R, Minv, ρg, η, dx, dy, γ) / (length(R.x) + length(R.y))
                push!(errs_in, err_in)
                println("\t", it_in ," inner iterations: error = ", err_in)
            end
            it_in += 1
        end
        compute_divV!(divV, V, dx, dy)
        err_out = norm(divV, 1) / length(divV)
        push!(errs_out, err_out)
        push!(itercounts, it_in)
        println(it_out , " outer iterations: error = ", err_out)
        it_out += 1
    end

    # println("outer iteration stopped at $it_out with error $err_out")

    return P, V, R, errs_in, errs_out, itercounts, xs, ys
end


function create_output_plot(P, V, R, errs_in, errs_out, itercounts, xs, ys; ncheck, η_ratio, savefig=false)
    dy = ys[2] - ys[1]
    nx = size(P, 1)
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations / nx", title="Mean Absolute Residual (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = cumsum(itercounts)
    iters_in  = ncheck .* (1:length(errs_in))
    scatter!(axs.err, iters_out ./ nx, log10.(errs_out), color=:blue, label="Pressure")
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:inferno),
           err=lines!(axs.err, iters_in ./ nx, log10.(errs_in), color=:red, label="Velocity"),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    axislegend(axs.err, position=:lb)

    if savefig
        save("2_output_$(η_ratio)_$(maximum(itercounts)).png", fig)
    else
        display(fig)
    end
    return nothing
end



function create_convergence_plot(errs_in, errs_out, itercounts; ncheck, η_ratio, nx, savefig=false)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Iterations / nx", ylabel="log₁₀(Mean Abs. Residual)", title="η ratio=$η_ratio")
    iters_out = cumsum(itercounts)
    iters_in  = ncheck .* (1:length(errs_in))
    lines!(ax, iters_in ./ nx, log10.(errs_in), color=:red, label="Velocity")
    scatter!(ax, iters_out ./ nx, log10.(errs_out), color=:blue, label="Pressure")
    axislegend(ax, position=:rt)
    if savefig
        save("2_convergence_$(η_ratio)_$(maximum(itercounts)).png", fig)
    else
        display(fig)
    end
    return nothing
end

eta_outer = 1e-4
eta_inner = 1.
n     = 127
ninner=1000
nouter=1
ncheck=10

outfields = linearStokes2D(n=n,
                           η_in=eta_inner, η_out=eta_outer, ρg_in=eta_outer,
                           niter_in=ninner, niter_out=nouter, ncheck=ncheck,
                           γ_factor=0.1,
                           max_err=1e-9)

create_output_plot(outfields...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, savefig=false)

create_convergence_plot(outfields[4:6]...; ncheck=ncheck, η_ratio=eta_inner/eta_outer, nx=n, savefig=false)
