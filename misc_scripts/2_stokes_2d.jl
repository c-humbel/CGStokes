# Step 1: linear Stokes solver in 2D with variationally consistent residual
# and variational boundary conditions
# and no-slip bedrock boundary 
using CairoMakie, Enzyme, LinearAlgebra, Printf

@views function J(V, P, P_old, ∇V, τ, ε̇, A, η, ρg, ω_air, ω_bed, γ, dx, dy)
    @. η.c[2:end-1, 2:end-1] = 0.5 * A.c^(-1)
    @. η.c[[1, end], :] = η.c[[2, end - 1], :]
    @. η.c[:, [1, end]] = η.c[:, [2, end - 1]]
    @. η.v = 0.5 * A.v^(-1)

    # compute velocity divergence
    @. ∇V.c = (V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx +
              (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy

    @. ∇V.v = (V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx +
              (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy

    @. ∇V.c[2:end-1, 2:end-1] *= ω_air.vc.x[2:end, 2:end-1] * ω_air.vc.x[1:end-1, 2:end-1] * ω_air.cv.y[2:end-1, 2:end] * ω_air.cv.y[2:end-1, 1:end-1]
    @. ∇V.v[2:end-1, 2:end-1] *= ω_air.cv.x[2:end, 2:end-1] * ω_air.cv.x[1:end-1, 2:end-1] * ω_air.vc.y[2:end-1, 2:end] * ω_air.vc.y[2:end-1, 1:end-1]
    @. ∇V.c[2:end-1, [1, end]] *= ω_air.vc.x[2:end, [1, end]] * ω_air.vc.x[1:end-1, [1, end]] * ω_air.cv.y[2:end-1, [1, end]]
    @. ∇V.v[2:end-1, [1, end]] *= ω_air.cv.x[2:end, [1, end]] * ω_air.cv.x[1:end-1, [1, end]] * ω_air.vc.y[2:end-1, [1, end]]
    @. ∇V.c[[1, end], 2:end-1] *= ω_air.vc.x[[1, end], 2:end-1] * ω_air.cv.y[[1, end], 2:end] * ω_air.cv.y[[1, end], 1:end-1]
    @. ∇V.v[[1, end], 2:end-1] *= ω_air.cv.x[[1, end], 2:end-1] * ω_air.vc.y[[1, end], 2:end] * ω_air.vc.y[[1, end], 1:end-1]
    @. ∇V.c[[1, end], [1, end]] *= ω_air.vc.x[[1, end], [1, end]] * ω_air.cv.y[[1, end], [1, end]] 
    @. ∇V.v[[1, end], [1, end]] *= ω_air.cv.x[[1, end], [1, end]] * ω_air.vc.y[[1, end], [1, end]] 

    # compute pressure
    @. P.c = P_old.c - γ * ∇V.c
    @. P.v = P_old.v - γ * ∇V.v

    # compute deviatoric strain rates
    @. ε̇.c.xx = (V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx
    @. ε̇.c.yy = (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy
    @. ε̇.c.xy[2:end-1, 2:end-1] = 0.5 * ((V.cv.x[2:end-1, 2:end] - V.cv.x[2:end-1, 1:end-1]) / dy +
                                          (V.vc.y[2:end, 2:end-1] - V.vc.y[1:end-1, 2:end-1]) / dx)

    @. ε̇.v.xx = (V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx
    @. ε̇.v.yy = (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy
    @. ε̇.v.xy[2:end-1, 2:end-1] = 0.5 * ((V.vc.x[2:end-1, 2:end] - V.vc.x[2:end-1, 1:end-1]) / dy +
                                          (V.cv.y[2:end, 2:end-1] - V.cv.y[1:end-1, 2:end-1]) / dx)

    # compute deviatoric stress
    @. τ.c.xx = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.xx
    @. τ.c.yy = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.yy
    @. τ.c.xy[2:end-1, 2:end-1] = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.xy[2:end-1, 2:end-1]

    @. τ.v.xx = 2 * η.v * ε̇.v.xx
    @. τ.v.yy = 2 * η.v * ε̇.v.yy
    @. τ.v.xy[2:end-1, 2:end-1] = 2 * η.v[2:end-1, 2:end-1] * ε̇.v.xy[2:end-1, 2:end-1]

    return 0.5 * (sum(τ.c.xx .* ε̇.c.xx .* ω_air.c[2:end-1, 2:end-1]) + sum(τ.c.yy .* ε̇.c.yy .* ω_air.c[2:end-1, 2:end-1])) + sum(τ.c.xy .* ε̇.c.xy .* ω_air.c) +
           0.5 * (sum(τ.v.xx .* ε̇.v.xx .* ω_air.v) + sum(τ.v.yy .* ε̇.v.yy .* ω_air.v)) + sum(τ.v.xy .* ε̇.v.xy .* ω_air.v) -
           sum((P_old.c .- 0.5γ .* ∇V.c) .* ∇V.c .* ω_air.c[2:end-1, 2:end-1]) - sum((P_old.v .- 0.5γ .* ∇V.v) .* ∇V.v .* ω_air.v) +
           sum(ρg.vc.x .* V.vc.x[2:end-1, :] .* ω_air.vc.x) + sum(ρg.vc.y .* V.vc.y[:, 2:end-1] .* ω_air.vc.y) +
           sum(ρg.cv.x .* V.cv.x[2:end-1, :] .* ω_air.cv.x) + sum(ρg.cv.y .* V.cv.y[:, 2:end-1] .* ω_air.cv.y)
end

@views function residual!(R, V, P, P_old, ∇V, τ, ε̇, A, η, ρg, ω_air, ω_bed, γ, dx, dy)
    # compute effective viscosity
    @. η.c[2:end-1, 2:end-1] = 0.5 * A.c^(-1)
    @. η.c[[1, end], :] = η.c[[2, end - 1], :]
    @. η.c[:, [1, end]] = η.c[:, [2, end - 1]]
    @. η.v = 0.5 * A.v^(-1)

    V.vc.x[ω_bed.vc.x .== 0] .= 0.
    V.cv.x[ω_bed.cv.x .== 0] .= 0.
    V.cv.y[ω_bed.cv.y .== 0] .= 0.
    V.vc.y[ω_bed.vc.y .== 0] .= 0.

    # compute velocity divergence
    @. ∇V.c = (V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx +
              (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy

    @. ∇V.v = (V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx +
              (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy

    @. ∇V.c[2:end-1, 2:end-1] *= ω_air.vc.x[2:end, 2:end-1] * ω_air.vc.x[1:end-1, 2:end-1] * ω_air.cv.y[2:end-1, 2:end] * ω_air.cv.y[2:end-1, 1:end-1]
    @. ∇V.v[2:end-1, 2:end-1] *= ω_air.cv.x[2:end, 2:end-1] * ω_air.cv.x[1:end-1, 2:end-1] * ω_air.vc.y[2:end-1, 2:end] * ω_air.vc.y[2:end-1, 1:end-1]
    @. ∇V.c[2:end-1, [1, end]] *= ω_air.vc.x[2:end, [1, end]] * ω_air.vc.x[1:end-1, [1, end]] * ω_air.cv.y[2:end-1, [1, end]]
    @. ∇V.v[2:end-1, [1, end]] *= ω_air.cv.x[2:end, [1, end]] * ω_air.cv.x[1:end-1, [1, end]] * ω_air.vc.y[2:end-1, [1, end]]
    @. ∇V.c[[1, end], 2:end-1] *= ω_air.vc.x[[1, end], 2:end-1] * ω_air.cv.y[[1, end], 2:end] * ω_air.cv.y[[1, end], 1:end-1]
    @. ∇V.v[[1, end], 2:end-1] *= ω_air.cv.x[[1, end], 2:end-1] * ω_air.vc.y[[1, end], 2:end] * ω_air.vc.y[[1, end], 1:end-1]
    @. ∇V.c[[1, end], [1, end]] *= ω_air.vc.x[[1, end], [1, end]] * ω_air.cv.y[[1, end], [1, end]] 
    @. ∇V.v[[1, end], [1, end]] *= ω_air.cv.x[[1, end], [1, end]] * ω_air.vc.y[[1, end], [1, end]] 

    # compute pressure
    @. P.c = P_old.c - γ * ∇V.c
    @. P.v = P_old.v - γ * ∇V.v

    # compute deviatoric strain rates
    @. ε̇.c.xx = (V.vc.x[2:end, :] - V.vc.x[1:end-1, :]) / dx
    @. ε̇.c.yy = (V.cv.y[:, 2:end] - V.cv.y[:, 1:end-1]) / dy
    @. ε̇.c.xy[2:end-1, 2:end-1] = 0.5 * ((V.cv.x[2:end-1, 2:end] - V.cv.x[2:end-1, 1:end-1]) / dy +
                                          (V.vc.y[2:end, 2:end-1] - V.vc.y[1:end-1, 2:end-1]) / dx)

    @. ε̇.v.xx = (V.cv.x[2:end, :] - V.cv.x[1:end-1, :]) / dx
    @. ε̇.v.yy = (V.vc.y[:, 2:end] - V.vc.y[:, 1:end-1]) / dy
    @. ε̇.v.xy[2:end-1, 2:end-1] = 0.5 * ((V.vc.x[2:end-1, 2:end] - V.vc.x[2:end-1, 1:end-1]) / dy +
                                          (V.cv.y[2:end, 2:end-1] - V.cv.y[1:end-1, 2:end-1]) / dx)

    # compute deviatoric stress
    @. τ.c.xx = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.xx
    @. τ.c.yy = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.yy
    @. τ.c.xy[2:end-1, 2:end-1] = 2 * η.c[2:end-1, 2:end-1] * ε̇.c.xy[2:end-1, 2:end-1]

    @. τ.v.xx = 2 * η.v * ε̇.v.xx
    @. τ.v.yy = 2 * η.v * ε̇.v.yy
    @. τ.v.xy[2:end-1, 2:end-1] = 2 * η.v[2:end-1, 2:end-1] * ε̇.v.xy[2:end-1, 2:end-1]

    @. R.vc.x = (P.c[2:end, :] * ω_air.c[3:end-1, 2:end-1] - P.c[1:end-1, :] * ω_air.c[2:end-2, 2:end-1]) / dx -
                (τ.c.xx[2:end, :] * ω_air.c[3:end-1, 2:end-1] - τ.c.xx[1:end-1, :] * ω_air.c[2:end-2, 2:end-1]) / dx -
                (τ.v.xy[2:end-1, 2:end] * ω_air.v[2:end-1, 2:end] - τ.v.xy[2:end-1, 1:end-1] * ω_air.v[2:end-1, 1:end-1]) / dy +
                ρg.vc.x * ω_air.vc.x
    @. R.cv.y = (P.c[:, 2:end] * ω_air.c[2:end-1, 3:end-1] - P.c[:, 1:end-1] * ω_air.c[2:end-1, 2:end-2]) / dy -
                (τ.c.yy[:, 2:end] * ω_air.c[2:end-1, 3:end-1] - τ.c.yy[:, 1:end-1] * ω_air.c[2:end-1, 2:end-2]) / dy -
                (τ.v.xy[2:end, 2:end-1] * ω_air.v[2:end, 2:end-1] - τ.v.xy[1:end-1, 2:end-1] * ω_air.v[1:end-1, 2:end-1]) / dx +
                ρg.cv.y * ω_air.cv.y

    @. R.cv.x = (P.v[2:end, :] * ω_air.v[2:end, :] - P.v[1:end-1, :] * ω_air.v[1:end-1, :]) / dx -
                (τ.v.xx[2:end, :] * ω_air.v[2:end, :] - τ.v.xx[1:end-1, :] * ω_air.v[1:end-1, :]) / dx -
                (τ.c.xy[2:end-1, 2:end] * ω_air.c[2:end-1, 2:end] - τ.c.xy[2:end-1, 1:end-1] * ω_air.c[2:end-1, 1:end-1]) / dy +
                ρg.cv.x * ω_air.cv.x

    @. R.vc.y = (P.v[:, 2:end] * ω_air.v[:, 2:end] - P.v[:, 1:end-1] * ω_air.v[:, 1:end-1]) / dy -
                (τ.v.yy[:, 2:end] * ω_air.v[:, 2:end] - τ.v.yy[:, 1:end-1] * ω_air.v[:, 1:end-1]) / dy -
                (τ.c.xy[2:end, 2:end-1] * ω_air.c[2:end, 2:end-1] - τ.c.xy[1:end-1, 2:end-1] * ω_air.c[1:end-1, 2:end-1]) / dx +
                ρg.vc.y * ω_air.vc.y

    @. R.vc.x *= ω_bed.vc.x[2:end-1, :]
    @. R.cv.y *= ω_bed.cv.y[:, 2:end-1]
    @. R.cv.x *= ω_bed.cv.x[2:end-1, :]
    @. R.vc.y *= ω_bed.vc.y[:, 2:end-1]

    return
end

@views function assign!(dest, src)
    for loc in eachindex(dest)
        for dir in eachindex(dest[loc])
            dest[loc][dir] .= src[loc][dir]
        end
    end
end

function dot_product(x, y)
    sum = 0.0
    for loc in eachindex(x)
        for dir in eachindex(x[loc])
            sum += dot(x[loc][dir], y[loc][dir])
        end
    end
    return sum
end

@views function apply_preconditioner(Z, R, A, η, ω, dx, dy, γ)
    # apply preconditioner to the residual
    # bc are included using ghost cells on η.c (what about η.v?)
    @. Z.vc.x = R.vc.x / (2 * (η.c[2:end-2, 2:end-1] + η.c[3:end-1, 2:end-1]) / dx^2
                          + (η.v[2:end-1, 1:end-1] + η.v[2:end-1, 2:end]) / dy^2
                          + 2γ / dx^2)

    @. Z.vc.y = R.vc.y / (2 * (η.v[:, 1:end-1] + η.v[:, 2:end]) / dy^2
                          + (η.c[1:end-1, 2:end-1] + η.c[2:end, 2:end-1]) / dx^2
                          + 2γ / dy^2)

    @. Z.cv.x = R.cv.x / (2 * (η.v[1:end-1, :] + η.v[2:end, :]) / dx^2
                          + (η.c[2:end-1, 1:end-1] + η.c[2:end-1, 2:end]) / dy^2
                          + 2γ / dx^2)

    @. Z.cv.y = R.cv.y / (2 * (η.c[2:end-1, 2:end-2] + η.c[2:end-1, 3:end-1]) / dy^2
                          + (η.v[1:end-1, 2:end-1] + η.v[2:end, 2:end-1]) / dx^2
                          + 2γ / dy^2)
end

@views function line_search(R, R̄, V, V̄, P, P̄, P_old, ∇V, ∇V̄, τ, τ̄, ε̇, ε̄, D, A, η, ρg, ω_air, ω_bed, γ, δ, dx, dy)
    make_zero!(R̄)
    make_zero!(V̄)
    make_zero!(P̄)
    make_zero!(τ̄)
    make_zero!(ε̄)
    make_zero!(∇V̄)

    @. V̄.vc.x[2:end-1, :] = D.vc.x
    @. V̄.vc.y[:, 2:end-1] = D.vc.y
    @. V̄.cv.x[2:end-1, :] = D.cv.x
    @. V̄.cv.y[:, 2:end-1] = D.cv.y

    autodiff(set_runtime_activity(Forward),
             residual!,
             DuplicatedNoNeed(R, R̄),
             Duplicated(V, V̄),
             Duplicated(P, P̄),
             Const(P_old),
             Duplicated(∇V, ∇V̄),
             Duplicated(τ, τ̄),
             Duplicated(ε̇, ε̄),
             Const(A), Const(η), Const(ρg), Const(ω_air), Const(ω_bed), Const(γ), Const(dx), Const(dy))

    return -δ / dot_product(D, R̄)
end

function smoothincf(x, y, xi, yi, ri, w, Ai, Ab)
    r = sqrt((x - xi)^2 + (y - yi)^2) - ri
    return 0.5 * (tanh(r/w) + 1)
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
    # free surface
    xf = 0.0lx
    yf = 1.2ly
    rf = 1.1ly
    # bedrock
    xb = 0.0lx
    yb = 1.2ly
    rb = 1.6ly
    # numerics
    nx, ny = 100, 100
    maxiter = 100nx
    ncheck = 1nx
    abstol = 1e-6
    maxiter_ph = 50
    # PH params
    γ = 1.0e1
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xv     = LinRange(-lx / 2, lx / 2, nx + 1)
    yv     = LinRange(-ly / 2, ly / 2, ny + 1)
    xc     = 0.5 .* (xv[1:end-1] .+ xv[2:end])
    yc     = 0.5 .* (yv[1:end-1] .+ yv[2:end])
    # arrays
    # volume fractions
    ω_air = (c  = zeros(nx + 2, ny + 2), v = zeros(nx + 1, ny + 1),
             vc = (x=zeros(nx - 1, ny),  y=zeros(nx + 1, ny)),
             cv = (x=zeros(nx, ny + 1),  y=zeros(nx, ny - 1)))
    ω_bed = (c  = zeros(nx + 2, ny + 2), v = zeros(nx + 1, ny + 1),
             vc = (x=zeros(nx + 1, ny), y=zeros(nx + 1, ny + 2)),
             cv = (x=zeros(nx + 2, ny + 1), y=zeros(nx, ny + 1)))
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
    ε̇ = (c=(xx=zeros(nx, ny),
             yy=zeros(nx, ny),
             xy=zeros(nx + 2, ny + 2)),
          v=(xx=zeros(nx + 1, ny + 1),
             yy=zeros(nx + 1, ny + 1),
             xy=zeros(nx + 1, ny + 1)))
    # velocity
    V = (vc=(x=zeros(nx + 1, ny), y=zeros(nx + 1, ny + 2)),
         cv=(x=zeros(nx + 2, ny + 1), y=zeros(nx, ny + 1)))
    # viscosity
    η = (c=zeros(nx + 2, ny + 2), v=zeros(nx + 1, ny + 1))
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
    # velocity divergence
    ∇V = (c=zeros(nx, ny), v=zeros(nx + 1, ny + 1))
    #  shadows
    R̄  = make_zero(R)
    V̄  = make_zero(V)
    P̄  = make_zero(P)
    τ̄  = make_zero(τ)
    ε̄  = make_zero(ε̇)
    ∇V̄ = make_zero(∇V)
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
    
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.c[2:end-1, 2:end-1], xc, yc')
    @. ω_air.c[[1, end], :] = ω_air.c[[2, end - 1], :]
    @. ω_air.c[:, [1, end]] = ω_air.c[:, [2, end - 1]]
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.v, xv, yv')
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.vc.x, xv[2:end-1], yc')
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.cv.y, xc, yv[2:end-1]')
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.cv.x, xc, yv')
    broadcast!((x, y) -> incf(x, y, xf, yf, rf, 0.0, 1.0), ω_air.vc.y, xv, yc')

    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.c[2:end-1, 2:end-1], xc, yc')
    @. ω_bed.c[[1, end], :] = ω_bed.c[[2, end - 1], :]
    @. ω_bed.c[:, [1, end]] = ω_bed.c[:, [2, end - 1]]
    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.v, xv, yv')
    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.vc.x, xv, yc')
    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.cv.y, xc, yv')
    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.cv.x[2:end-1, :], xc, yv')
    broadcast!((x, y) -> incf(x, y, xb, yb, rb, 1.0, 0.0), ω_bed.vc.y[:, 2:end-1], xv, yc')
    @. ω_bed.cv.x[[1, end], :] = ω_bed.cv.x[[2, end-1], :]
    @. ω_bed.vc.y[:, [1, end]] = ω_bed.vc.y[:, [1, end]]
    # plots
    fig = Figure(; size=(600, 450))
    axs = (Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="A"),
           Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="P"),
           Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="ρg"),
           Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="ω"))
    hms = (heatmap!(axs[1], xc, yc, A.c; colormap=:roma),
           heatmap!(axs[2], xc, yc, P.c; colormap=:turbo),
           heatmap!(axs[3], xv, yc, ρg.vc.y; colormap=:viridis),
           heatmap!(axs[4], xc, yc, ω_air.c .* ω_bed.c; colormap=Makie.Reverse(:grays)))
    cbs = (Colorbar(fig[1, 1][1, 2], hms[1]),
           Colorbar(fig[2, 1][1, 2], hms[2]),
           Colorbar(fig[1, 2][1, 2], hms[3]),
           Colorbar(fig[2, 2][1, 2], hms[4]))
    display(fig)
    # Powell-Hestenes pressure solver
    for iter_ph in 1:maxiter_ph
        P_old.c .= P.c
        P_old.v .= P.v
        # velocity solver
        # init residual
        residual!(R, V, P, P_old, ∇V, τ, ε̇, A, η, ρg, ω_air, ω_bed, γ, dx, dy)
        apply_preconditioner(Z, R, A, η, ω_air, dx, dy, γ)
        # init search direction
        assign!(D, Z)
        δ = dot_product(R, Z)
        # CG iterative loop
        for iter in 1:maxiter
            α = line_search(R, R̄, V, V̄, P, P̄, P_old, ∇V, ∇V̄, τ, τ̄, ε̇, ε̄, D, A, η, ρg, ω_air, ω_bed, γ, δ, dx, dy)

            @. V.vc.x[2:end-1, :] += α * D.vc.x
            @. V.vc.y[:, 2:end-1] += α * D.vc.y
            @. V.cv.x[2:end-1, :] += α * D.cv.x
            @. V.cv.y[:, 2:end-1] += α * D.cv.y

            residual!(R, V, P, P_old, ∇V, τ, ε̇, A, η, ρg, ω_air, ω_bed, γ, dx, dy)
            apply_preconditioner(Z, R, A, η, ω_air, dx, dy, γ)

            δ_new = dot_product(R, Z)
            β = δ_new / δ
            δ = δ_new

            @. D.vc.x = β * D.vc.x + Z.vc.x
            @. D.vc.y = β * D.vc.y + Z.vc.y
            @. D.cv.x = β * D.cv.x + Z.cv.x
            @. D.cv.y = β * D.cv.y + Z.cv.y

            if iter % ncheck == 0
                err = (maximum(abs.(R.vc.x)),
                       maximum(abs.(R.cv.y)),
                       maximum(abs.(R.cv.x)),
                       maximum(abs.(R.vc.y)))
                @printf("    iter  = %.1f × N, err = [%1.3e, %1.3e, %1.3e, %1.3e]\n", iter / nx, err...)
                if any(!isfinite, err)
                    error("simulation failed")
                end
                if all(err .< abstol)
                    break
                end
            end
        end

        err_Pr = (maximum(abs.(∇V.c .* ω_air.c[2:end-1, 2:end-1])),
                  maximum(abs.(∇V.v .* ω_air.v)))

        @printf("iter_ph = %d, err_Pr = [%1.3e, %1.3e]\n", iter_ph, err_Pr...)

        if any(!isfinite, err_Pr)
            error("simulation failed")
        end
        if all(err_Pr .< abstol)
            break
        end

        hms[2][3] = P.c #.* ω_air.c[2:end-1, 2:end-1]
        hms[3][3] = V.vc.y[:, 2:end-1] # .* ω_air.vc.y
        display(fig)
    end

    println("Check variational consistency")
    residual!(R, V, P, P_old, ∇V, τ, ε̇, A, η, ρg, ω_air, ω_bed, γ, dx, dy)

    make_zero!(V̄)
    make_zero!(P̄)
    make_zero!(τ̄)
    make_zero!(ε̄)
    make_zero!(∇V̄)

    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), J,
                    Active,
                    Duplicated(V, V̄),
                    Duplicated(P, P̄),
                    Const(P_old),
                    Duplicated(∇V, ∇V̄),
                    Duplicated(τ, τ̄),
                    Duplicated(ε̇, ε̄),
                    Const(A), Const(η), Const(ρg), Const(ω_air), Const(ω_bed), Const(γ), Const(dx), Const(dy))

    vcheck = (maximum(abs.(V̄.vc.x[2:end-1, :] .- R.vc.x).*ω_bed.vc.x[2:end-1, :]),
              maximum(abs.(V̄.cv.y[:, 2:end-1] .- R.cv.y).*ω_bed.cv.y[:, 2:end-1]),
              maximum(abs.(V̄.cv.x[2:end-1, :] .- R.cv.x).*ω_bed.cv.x[2:end-1, :]),
              maximum(abs.(V̄.vc.y[:, 2:end-1] .- R.vc.y).*ω_bed.vc.y[:, 2:end-1]))
    @show vcheck
end

main()