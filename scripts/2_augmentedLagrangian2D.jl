using CairoMakie
using LinearAlgebra
using Enzyme


function compute_P!(P, P_old, V, dx, dy, γ)
    nx, ny = size(P)
    # compute pressure
    for j = 1:ny
        for i = 1:nx
            dVx  = (V.x[i+1, j] - V.x[i, j]) / dx
            dVy  = (V.y[i, j+1] - V.y[i, j]) / dy
            divV = dVx + dVy
            P[i, j]  = P_old[i, j] - γ * divV
        end
    end
    return nothing
end


function compute_R!(R, P, V, ρg, η, dx, dy, γ)
    nx, ny = size(P)
    # compute the residual at cell interfaces

    # in horizontal (x) direction
    for j = 1:ny-2
        for i = 1:nx-1
            # stress at horizontally adjacent cell centers
            τxx_r = (2 * η[i+1, j+1] - γ) * (V.x[i+2, j+1] - V.x[i+1, j+1]) / dx - γ * (V.y[i+1, j+2] - V.y[i+1, j+1]) / dy
            τxx_l = (2 * η[i  , j+1] - γ) * (V.x[i+1, j+1] - V.x[i  , j+1]) / dx - γ * (V.y[i  , j+2] - V.y[i  , j+1]) / dy
            # viscosity at vertically adjacent cell corners
            η_t   = 0.25 * (η[i, j+1] + η[i+1, j+1] + η[i, j+2] + η[i+1, j+2])
            η_b   = 0.25 * (η[i, j  ] + η[i+1, j  ] + η[i, j+1] + η[i+1, j+1])
            # stress at same cell corners
            τxy_t = η_t * ((V.x[i+1, j+2] - V.x[i+1, j+1]) / dy
                         + (V.y[i+1, j+2] - V.y[i  , j+2]) / dx)
            τxy_b = η_b * ((V.x[i+1, j+1] - V.x[i+1, j  ]) / dy
                         + (V.y[i+1, j+1] - V.y[i  , j+1]) / dx)
            # residual in x direction on the interface
            R.x[i, j]  = ((τxx_r - τxx_l) / dx
                        + (τxy_t - τxy_b) / dy
                        - (P[i+1, j+1] - P[i, j+1]) / dx)
        end
    end

    # residual in y direction
    for j = 1:ny-1
        for i = 1:nx-2
            τyy_t = (2 * η[i+1, j+1] - γ) * (V.y[i+1, j+2] - V.y[i+1, j+1]) / dy - γ * (V.x[i+2, j+1] - V.x[i+1, j+1]) / dx
            τyy_b = (2 * η[i+1, j  ] - γ) * (V.y[i+1, j+1] - V.y[i+1, j  ]) / dy - γ * (V.x[i+2, j  ] - V.x[i+1, j  ]) / dx

            η_r   = 0.25 * (η[i+1, j  ] + η[i+2, j  ] + η[i+1, j+1] + η[i+2, j+1])
            η_l   = 0.25 * (η[i  , j  ] + η[i+1, j  ] + η[i  , j+1] + η[i+1, j+1])

            τxy_r = η_r * ((V.x[i+2, j+1] - V.x[i+2, j  ]) / dy
                         + (V.y[i+2, j+1] - V.y[i+1, j+1]) / dx)
            τxy_l = η_l * ((V.x[i+1, j+1] - V.x[i+1, j  ]) / dy
                         + (V.y[i+1, j+1] - V.y[i  , j+1]) / dx)
            
            R.y[i, j]  = ((τyy_t - τyy_b) / dy
                        + (τxy_r - τxy_l) / dx
                        - (P[i+1, j+1] - P[i+1, j]) / dy
                        + (ρg[i+1, j+1] + ρg[i+1, j]) * 0.5)
        end
    end
    return nothing
end


function compute_β_normR(normR², R)
    normR²_new = dot(R.x, R.x) + dot(R.y, R.y)
    return normR²_new / normR², normR²_new
end


function update_D!(D, R, Minv, β)
    for j = 2:size(D.x, 2)-1
        for i = 2:size(D.x, 1)-1
            D.x[i, j] = Minv.x[i-1, j-1] * R.x[i-1, j-1] + β * D.x[i, j]
        end
    end

    for j = 2:size(D.y, 2)-1
        for i = 2:size(D.y, 1)-1
            D.y[i, j] = Minv.y[i-1, j-1] * R.y[i-1, j-1] + β * D.y[i, j]
        end
    end
    return nothing
end


function compute_α(R, Ad, P, V, D, ρg, η, normR², dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Ad
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Ad), Const(P), Duplicated(V, D), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = -dot(R, R) / dot(D, A*D)
    # negative sign because matrix A produces the negative divergenve of stress 
    return  -normR² / (dot(D.x[2:end-1, 2:end-1], Ad.x) + dot(D.y[2:end-1, 2:end-1], Ad.y))
end


function update_V!(V, D, Minv, α)
    for j = 2:size(V.x, 2)-1
        for i = 2:size(V.x, 1)-1
            V.x[i, j] += α * Minv.x[i-1, j-1] * D.x[i, j]
        end
    end
    for j = 2:size(V.y, 2)-1
        for i = 2:size(V.y, 1)-1
            V.y[i, j] += α * Minv.y[i-1, j-1] * D.y[i, j]
        end
    end
    return nothing
end


function neumann_bc_x!(f)
    for j = axes(f, 2)
        f[1,   j] = f[2,     j]
        f[end, j] = f[end-1, j]
    end
    return nothing
end

function neumann_bc_y!(f)
    for i = axes(f, 1)
        f[i, 1  ] = f[i, 2    ]
        f[i, end] = f[i, end-1]
    end
    return nothing
end


function linearStokes2D(η_ratio=0.1; max_iter=100000, ncheck=2000, max_err=1e-6, n=127)
    Lx = Ly = 10.
    R_in  = 1.
    η_out = 1.
    η_in  = η_ratio * η_out
    ρg_in = 1.

    nx = ny = n

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # field initialisation
    η      = [x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xs, y=ys]
    ρg     = [x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xs, y=ys]
    P      = zeros(nx, ny)
    P_old  = zeros(nx, ny)
    V      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1))
    D      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1))  # search direction of CG, outer cells are zero
    R      = (x =zeros(nx-1, ny-2), y =zeros(nx-2, ny-1))  # Residuals of velocity PDE
    Ad     = (x =zeros(nx-1, ny-2), y =zeros(nx-2, ny-1))  # Jacobian of compute_R wrt. V, multiplied by search vector D

    # preconditioner
    preθv = min(dx, dy)^2 / 4.1
    Minv = (x=[preθv / max(η[i, j], η[i+1, j], η[i, j+1], η[i+1, j+1], η[i, j+2], η[i+1, j+2])
                for i=1:nx-1, j=1:ny-2],
            y=[preθv / max(η[i, j], η[i+1, j], η[i+2, j], η[i, j+1], η[i+1, j+1], η[i+2, j+1])
                for i=1:nx-2, j=1:ny-1])

    # Coefficient of augmented Lagrangian
    γ = inv(maximum(η))
    
    # visualisation
    errs = []

    err_out = 2 * max_err
    it_out = 1
    # outer loop, Powell Hestenes
    while it_out <= 1 && err_out > max_err
        err_in = 2 * max_err
        it_in = 1
        # inner loop, Conjugate Gradient
        # iteration zero
        compute_R!(R, P, V, ρg, η, dx, dy, γ)
        D.x[2:end-1, 2:end-1] .= Minv.x .* R.x
        D.y[2:end-1, 2:end-1] .= Minv.y .* R.y
        normR² = dot(R.x, R.x) + dot(R.y, R.y)
        while it_in <= max_iter && err_in > max_err
            α = compute_α(R, Ad, P, V, D, ρg, η, normR², dx, dy, γ)
            update_V!(V, D, Minv, α)
            neumann_bc_y!(V.x)
            neumann_bc_x!(V.y)
            compute_R!(R, P, V, ρg, η, dx, dy, γ)
            β, normR² = compute_β_normR(normR², R)
            update_D!(D, R, Minv, β)
            err_in = sqrt(normR²) / (length(R.x) + length(R.y))
            if it_in % ncheck == 0
                push!(errs, err_in)
                println(it_in ," inner iterations: error = ", err_in)
            end
            it_in += 1
        end
        compute_P!(P, P_old, V, dx, dy, γ)
        # err_out = sqrt(norm∇V²) / length(P)
        # push!(errs, err_out)
        # println(it_out , " outer iterations: performed ", it_in ," inner iterations, error = ", err_out)
        it_out += 1
    end

    # println("outer iteration stopped at $it_out with error $err_out")

    return P, V, R, errs, xs, ys
end


function create_output_plot(P, V, R, errs, xs, ys, ncheck, η_ratio; savefig=false)
    dy = ys[2] - ys[1]
    fig = Figure()
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations", title="Residual (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:inferno),
           err=lines!(axs.err, ncheck .* 1:length(errs), log10.(errs)),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)

    if savefig
        save("1_output_$(η_ratio)_$(n).png", fig)
    else
        display(fig)
    end
    return nothing
end


function create_convergence_plot(errs, ncheck, η_ratio; savefig=false)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Iterations", ylabel="log₁₀(Residual)", title="η ratio=$η_ratio")
    scatterlines!(ax, ncheck .* (1:length(errs)), log10.(errs))
    if savefig
        save("1_convergence_$(η_ratio)_$(n).png", fig)
    else
        display(fig)
    end
    return nothing
    
end

outfields = linearStokes2D(0.1; max_iter=500, ncheck=50)

create_output_plot(outfields..., 1, 0.1)