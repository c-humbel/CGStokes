using CairoMakie
using LinearAlgebra
using Enzyme


function compute_P!(P, divV, V, dx, dy, γ)
    nx, ny = size(P)
    # compute pressure
    for j = 1:ny
        for i = 1:nx
            divV[i, j] = (V.x[i+1, j] - V.x[i, j]) / dx + (V.y[i, j+1] - V.y[i, j]) / dy
            P[i, j] -= γ * divV[i, j]
        end
    end
    return nothing
end


function compute_R!(R, P, V, ρg, η, dx, dy, γ)
    nx, ny = size(P)
    # compute the residual at cell interfaces

    # in horizontal (x) direction
    for j = 2:ny-1
        for i = 2:nx
            # stress at horizontally adjacent cell centers
            τxx_r = (2 * η[i  , j] + γ) * (V.x[i+1, j] - V.x[i  , j]) / dx + γ * (V.y[i  , j+1] - V.y[i  , j]) / dy
            τxx_l = (2 * η[i-1, j] + γ) * (V.x[i  , j] - V.x[i-1, j]) / dx + γ * (V.y[i-1, j+1] - V.y[i-1, j]) / dy
            # viscosity at vertically adjacent cell corners
            η_t   = 0.25 * (η[i-1, j  ] + η[i, j  ] + η[i-1, j+1] + η[i, j+1])
            η_b   = 0.25 * (η[i-1, j-1] + η[i, j-1] + η[i-1, j  ] + η[i, j  ])
            # stress at same cell corners
            τxy_t = η_t * ((V.x[i, j+1] - V.x[i  , j  ]) / dy
                         + (V.y[i, j+1] - V.y[i-1, j+1]) / dx)
            τxy_b = η_b * ((V.x[i, j  ] - V.x[i  , j-1]) / dy
                         + (V.y[i, j  ] - V.y[i-1, j  ]) / dx)
            # residual in x direction on the interface
            R.x[i, j]  = (- (τxx_r - τxx_l) / dx
                          - (τxy_t - τxy_b) / dy
                          + (P[i, j] - P[i-1, j]) / dx)
        end
    end

    # residual in y direction
    for j = 2:ny
        for i = 2:nx-1
            τyy_t = (2 * η[i, j  ] + γ) * (V.y[i, j+1] - V.y[i, j  ]) / dy + γ * (V.x[i+1, j  ] - V.x[i, j  ]) / dx
            τyy_b = (2 * η[i, j-1] + γ) * (V.y[i, j  ] - V.y[i, j-1]) / dy + γ * (V.x[i+1, j-1] - V.x[i, j-1]) / dx

            η_r   = 0.25 * (η[i  , j-1] + η[i+1, j-1] + η[i  , j  ] + η[i+1, j  ])
            η_l   = 0.25 * (η[i-1, j-1] + η[i  , j-1] + η[i-1, j  ] + η[i  , j  ])

            τxy_r = η_r * ((V.x[i+1, j] - V.x[i+1, j-1]) / dy
                         + (V.y[i+1, j] - V.y[i  , j  ]) / dx)
            τxy_l = η_l * ((V.x[i  , j] - V.x[i  , j-1]) / dy
                         + (V.y[i  , j] - V.y[i-1, j  ]) / dx)
            
            R.y[i, j]  = ( - (τyy_t - τyy_b) / dy
                           - (τxy_r - τxy_l) / dx
                           + (P[i, j] - P[i, j-1]) / dy
                           - (ρg[i, j] + ρg[i, j-1]) * 0.5)
        end
    end

    # boundary conditions

    # wall normal velocities are zero
    for j = 1:ny
        R.x[1  , j] = -V.x[1  , j]
        R.x[end, j] = -V.x[end, j]
    end

    for i = 1:nx
        R.y[i, 1  ] = -V.y[i, 1  ]
        R.y[i, end] = -V.y[i, end]
    end

    # velocities parallel to wall should have no gradient in the wall normal direction
    for i = 2:nx
        R.x[i, 1  ] = -(V.x[i, 2    ] - V.x[i, 1  ]) / dy
        R.x[i, end] = -(V.x[i, end-1] - V.x[i, end]) / dy
    end

    for j = 2:ny
        R.y[1  , j] = -(V.y[2    , j] - V.y[1  , j]) / dx
        R.y[end, j] = -(V.y[end-1, j] - V.y[end, j]) / dx
    end

    return nothing
end


function compute_β_rMr(rMr, R, Minv)
    rMr_new = dot(R.x, Minv.x .* R.x) + dot(R.y, Minv.y .* R.y)
    return rMr_new / rMr, rMr_new
end


function update_D!(D, R, Minv, β)
    for j = axes(D.x, 2)
        for i =axes(D.x, 1)
            D.x[i, j] = Minv.x[i, j] * R.x[i, j] + β * D.x[i, j]
        end
    end

    for j = axes(D.y, 2)
        for i = axes(D.y, 1)
            D.y[i, j] = Minv.y[i, j] * R.y[i, j] + β * D.y[i, j]
        end
    end
    return nothing
end


function compute_α(R, Ad, P, V, D, ρg, η, rMr, dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Ad
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Ad), Const(P), Duplicated(V, D), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # negated because autodiff gives the ∂/∂V (b - A V) * D = - A*D
    return  rMr / (dot(D.x, -Ad.x) + dot(D.y, -Ad.y))
end


function update_V!(V, D, α)
    for j = axes(V.x, 2)
        for i = axes(V.x, 1)
            V.x[i, j] += α * D.x[i, j]
        end
    end
    for j = axes(V.y, 2)
        for i = axes(V.y, 1)
            V.y[i, j] += α * D.y[i, j]
        end
    end
    return nothing
end



function linearStokes2D(η_ratio=0.1; niter_in=1000, niter_out=1000, ncheck=2000, max_err=1e-6, n=127)
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
    η    = [x^2 + y^2 < R_in^2 ? η_in  : η_out for x=xs, y=ys]
    ρg   = [x^2 + y^2 < R_in^2 ? ρg_in : 0.    for x=xs, y=ys]
    P    = zeros(nx, ny)
    divV = zeros(nx, ny)
    V    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    D    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # search direction of CG, outer cells are zero
    R    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Residuals of velocity PDE
    Ad   = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))  # Jacobian of compute_R wrt. V, multiplied by search vector D
    Minv = (x= ones(nx+1, ny), y= ones(nx, ny+1))  # diagonal preconditioner

    # fill inner values of Minv
    preθv = min(dx, dy)^2 / 4.1
    for j = 2:ny-1
        for i = 2:nx
            Minv.x[i, j] = preθv /( max(η[i-1, j-1], η[i, j-1], η[i-1, j], η[i, j], η[i-1, j+1], η[i, j+1]) + γ)
        end
    end
    for j = 2:ny
        for i = 2:nx-1
            Minv.y[i, j] = preθv / (max(η[i-1, j-1], η[i, j-1], η[i+1, j-1], η[i-1, j], η[i, j], η[i+1, j]) + γ)
        end
    end

    # Coefficient of augmented Lagrangian
    γ = inv(maximum(η))
    
    # visualisation
    errs_in = []
    errs_out = []

    err_out = 2 * max_err
    it_out = 1
    # outer loop, Powell Hestenes
    while it_out <= niter_out && err_out > max_err
        err_in = 2 * max_err
        it_in = 1
        # inner loop, Conjugate Gradient
        # iteration zero
        compute_R!(R, P, V, ρg, η, dx, dy, γ)
        update_D!(D, R, Minv, 0.)
        rMr = dot(R.x, Minv.x .* R.x) + dot(R.y, Minv.y .* R.y)
        while it_in <= niter_in && err_in > max_err
            α = compute_α(R, Ad, P, V, D, ρg, η, rMr, dx, dy, γ)
            update_V!(V, D, α)
            compute_R!(R, P, V, ρg, η, dx, dy, γ)
            β, rMr = compute_β_rMr(rMr, R, Minv)
            update_D!(D, R, Minv, β)
            if it_in % ncheck == 0
                err_in = (norm(R.x, 1) + norm(R.y, 1)) / (length(R.x) + length(R.y))
                push!(errs_in, err_in)
                println("\t", it_in ," inner iterations: error = ", err_in)
            end
            it_in += 1
        end
        compute_P!(P, divV, V, dx, dy, γ)
        err_out = norm(divV, 1) / length(divV)
        push!(errs_out, err_out)
        println(it_out , " outer iterations: error = ", err_out)
        it_out += 1
    end

    # println("outer iteration stopped at $it_out with error $err_out")

    return P, V, R, errs_in, errs_out, xs, ys
end


function create_output_plot(P, V, R, errs_in, errs_out, xs, ys; ncheck, ninner, η_ratio, savefig=false)
    dy = ys[2] - ys[1]
    fig = Figure(size=(800, 600))
    axs = (P=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
        err=Axis(fig[1,2][1,1], xlabel="Iterations", title="Mean Absolute Residual (log)"),
        Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
        Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual (log)"))
    # compute location of outer iteration errors
    iters_out = [ninner * i for i=1:length(errs_out)]
    iters_out[end] = ncheck * length(errs_in)
    scatter!(axs.err, iters_out, log10.(errs_out), color=:blue, label="Pressure")
    plt = (P=image!(axs.P, (xs[1], xs[end]), (ys[1], ys[end]), P, colormap=:inferno),
           err=lines!(axs.err, ncheck .* (1:length(errs_in)), log10.(errs_in), color=:red, label="Velocity"),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[2], xs[end-1]), (ys[1]+dy/2, ys[end]-dy/2), log10.(abs.(R.y)), colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.P)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)
    axislegend(axs.err, position=:rt)

    if savefig
        save("2_output_$(η_ratio)_$(ninner).png", fig)
    else
        display(fig)
    end
    return nothing
end



function create_convergence_plot(errs_in, errs_out, ncheck, ninner, η_ratio; savefig=false)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Iterations", ylabel="log₁₀(Mean Abs. Residual)", title="η ratio=$η_ratio")
    iters_out = [ninner * i for i=1:length(errs_out)]
    iters_out[end] = ncheck * length(errs_in)
    scatterlines!(ax, ncheck .* (1:length(errs_in)), log10.(errs_in), color=:red, label="Velocity")
    #scatter!(ax, iters_out, log10.(errs_out), color=:blue, label="Pressure")
    #axislegend(ax, position=:rt)
    if savefig
        save("2_convergence_$(η_ratio)_$(ninner).png", fig)
    else
        display(fig)
    end
    return nothing
    
end

eta = 0.1
ninner=500
nouter=2
ncheck=10

outfields = linearStokes2D(eta; niter_in=ninner, niter_out=nouter, ncheck=ncheck)

create_output_plot(outfields...; ncheck=ncheck, ninner=ninner, η_ratio=eta, savefig=false)

create_convergence_plot(outfields[4:5]..., ncheck, ninner, eta; savefig=false)