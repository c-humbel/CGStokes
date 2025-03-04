using CairoMakie, Enzyme, LinearAlgebra

@views function J()
    return
end

@views function residual!(R, V, P, P_old, τ, A, η, ρg, γ, dx, dy)
    # compute effective viscosity
    @. η.c = 0.5 * A.c^(-1)
    @. η.v = 0.5 * A.v^(-1)

    # compute pressure
    @. P.c = P_old.c - γ * ((V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx +
                            (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy)

    @. P.v = P_old.v - γ * ((V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx +
                            (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy)

    # compute deviatoric stress
    @. τ.c.xx = 2 * η.c * (V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx
    @. τ.c.yy = 2 * η.c * (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy
    @. τ.c.xy[2:end-1, 2:end-1] = η.c * ((V.cv.x[2:end-1, 2:end] - V.cv.x[2:end-1, 1:end-1]) / dy +
                                         (V.vc.y[2:end, 2:end-1] - V.vc.y[1:end-1, 2:end-1]) / dx)

    @. τ.v.xx = 2 * η.v * (V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx
    @. τ.v.yy = 2 * η.v * (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy
    @. τ.v.xy[2:end-1, 2:end-1] = η.v[2:end-1, 2:end-1] * ((V.vc.x[2:end-1, 2:end] - V.vc.x[2:end-1, 1:end-1]) / dy +
                                                           (V.cv.y[2:end, 2:end-1] - V.cv.y[1:end-1, 2:end-1]) / dx)

    @. R.vc.x = (P.c[2:end, :] - P.c[1:end-1, :]) / dx -
                (τ.c.xx[2:end, :] - τ.c.xx[1:end-1, :]) / dx -
                (τ.v.xy[2:end-1, 2:end] - τ.v.xy[2:end-1, 1:end-1]) / dy +
                ρg.vc.x
    @. R.cv.y = (P.c[:, 2:end] - P.c[:, 1:end-1]) / dy -
                (τ.c.yy[:, 2:end] - τ.c.yy[:, 1:end-1]) / dy -
                (τ.v.xy[2:end, 2:end-1] - τ.v.xy[1:end-1, 2:end-1]) / dx +
                ρg.cv.y

    @. R.cv.x = (P.v[2:end, :] - P.v[1:end-1, :]) / dx -
                (τ.v.xx[2:end, :] - τ.v.xx[1:end-1, :]) / dx -
                (τ.c.xy[2:end-1, 2:end] - τ.c.xy[2:end-1, 1:end-1]) / dy +
                ρg.cv.x
    @. R.vc.y = (P.v[:, 2:end] - P.v[:, 1:end-1]) / dy -
                (τ.v.yy[:, 2:end] - τ.v.yy[:, 1:end-1]) / dy -
                (τ.c.xy[2:end, 2:end-1] - τ.c.xy[1:end-1, 2:end-1]) / dx +
                ρg.vc.y

    return
end

@views function preconditioner!(Z, R, η, dx, dy)
    Z.vc.x .= R.vc.x ./ (η.c[1:end-1, :] + η.c[2:end, :] + η.v[2:end-1, 1:end-1] + η.v[2:end-1, 2:end])
    Z.cv.y .= R.cv.y ./ (η.c[:, 1:end-1] + η.c[:, 2:end] + η.v[1:end-1, 2:end-1] + η.v[2:end, 2:end-1])

    Z.cv.x[:, 2:end-1]  .= R.cv.x[:, 2:end-1] ./ (η.v[1:end-1, 2:end-1] + η.v[2:end, 2:end-1] + η.c[:, 1:end-1] + η.c[:, 2:end])
    Z.cv.x[:, [1, end]] .= R.cv.x[:, [1, end]] ./ (η.v[1:end-1, [1, end]] + η.v[2:end, [1, end]])

    Z.vc.y[2:end-1, :]  .= R.vc.y[2:end-1, :] ./ (η.v[2:end-1, 1:end-1] + η.v[2:end-1, 2:end] + η.c[1:end-1, :] + η.c[2:end, :])
    Z.vc.y[[1, end], :] .= R.vc.y[[1, end], :] ./ (η.v[[1, end], 1:end-1] + η.v[[1, end], 2:end])

    return
end

@views function line_search!()
    # TODO
    return
end

@views function main()
    # physics
    lx, ly = 1.0, 1.0
    Ab     = 1.0
    ρgb    = 1.0
    # inclusions
    ri  = (0.1, 0.15, 0.2)
    xi  = (-0.3, 0.0, 0.2)
    yi  = (-0.3, 0.2, -0.2)
    Ai  = (0.1, 0.2, 0.3)
    ρgi = (0.0, 0.0, 0.0)
    # numerics
    nx, ny  = 50, 50
    maxiter = 100nx^2
    ncheck  = 1nx^2
    # PH params
    γ = 1.0
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xv     = LinRange(-lx / 2, lx / 2, nx + 1)
    yv     = LinRange(-ly / 2, ly / 2, ny + 1)
    xc     = 0.5 .* (xv[1:end-1] .+ xv[2:end])
    yc     = 0.5 .* (yv[1:end-1] .+ yv[2:end])
    # arrays
    # pressure
    P     = (c=zeros(nx, ny), v=zeros(nx + 1, ny + 1))
    P_old = (c=zeros(nx, ny), v=zeros(nx + 1, ny + 1))
    # deviatoric stress
    τ = (c=(xx=zeros(nx, ny),
            yy=zeros(nx, ny),
            xy=zeros(nx + 2, ny + 2)),
         v=(xx=zeros(nx + 1, ny + 1),
            yy=zeros(nx + 1, ny + 1),
            xy=zeros(nx + 1, ny + 1)))
    # velocity
    V = (vc=(x=zeros(nx + 1, ny), y=zeros(nx + 1, ny + 2)),
         cv=(x=zeros(nx + 2, ny + 1), y=zeros(nx, ny + 1)))
    # viscosity
    η = (c=zeros(nx, ny), v=zeros(nx + 1, ny + 1))
    A = (c=zeros(nx, ny), v=zeros(nx + 1, ny + 1))
    # gravity
    ρg = (vc=(x=zeros(nx - 1, ny), y=zeros(nx + 1, ny)),
          cv=(x=zeros(nx, ny + 1), y=zeros(nx, ny - 1)))
    # residual
    R = (vc = (x=zeros(nx - 1, ny), y=zeros(nx + 1, ny)),
         cv = (x=zeros(nx, ny + 1), y=zeros(nx, ny - 1)))
    # preconditioned residual
    Z = (vc=(x=zeros(nx - 1, ny), y=zeros(nx + 1, ny)),
         cv=(x=zeros(nx, ny + 1), y=zeros(nx, ny - 1)))
    # search direction
    D = (vc=(x=zeros(nx - 1, ny), y=zeros(nx + 1, ny)),
         cv=(x=zeros(nx, ny + 1), y=zeros(nx, ny - 1)))
    # initial conditions
    incf(x, y, xi, yi, ri, Ai, Ab) = (x - xi)^2 + (y - yi)^2 < ri^2 ? Ai : Ab
    A.c .= Ab
    A.v .= Ab
    ρg.vc.y .= ρgb
    ρg.cv.y .= ρgb
    for (_xi, _yi, _ri, Ai, ρgi) in zip(xi, yi, ri, Ai, ρgi)
        # flow parameter
        broadcast!((x, y, _A) -> incf(x, y, _xi, _yi, _ri, Ai, _A), A.c, xc, yc', A.c)
        broadcast!((x, y, _A) -> incf(x, y, _xi, _yi, _ri, Ai, _A), A.v, xv, yv', A.v)
        # gravity
        broadcast!((x, y, _ρg) -> incf(x, y, _xi, _yi, _ri, ρgi, _ρg), ρg.vc.y, xv, yc', ρg.vc.y)
        broadcast!((x, y, _ρg) -> incf(x, y, _xi, _yi, _ri, ρgi, _ρg), ρg.cv.y, xc, yv[2:end-1]', ρg.cv.y)
    end
    # plots
    fig = Figure(; size=(600, 400))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="A"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="P"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="ρg"))
    hms = (heatmap!(axs[1], xc, yc, A.c; colormap=:roma),
           heatmap!(axs[2], xc, yc, P.c; colormap=:turbo),
           heatmap!(axs[3], xv, yc, ρg.vc.y; colormap=:viridis))
    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[2, 1][1, 2], hms[2]),
           Colorbar(fig[1, 2][1, 2], hms[3]))
    display(fig)
    # velocity solver
    residual!(R, V, P, P_old, τ, A, η, ρg, γ, dx, dy)
    preconditioner!(Z, R, η, dx, dy)
    D.vc.x .= Z.vc.x
    D.cv.y .= Z.cv.y
    D.cv.x .= Z.cv.x
    D.vc.y .= Z.vc.y

    for iter in 1:maxiter
        # TODO
        break
    end
    return
end

main()
