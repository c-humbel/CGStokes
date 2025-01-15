using Enzyme
using Random

function compute_R!(R, P, η, P₀, V, ρg, B, q, dx, dy, γ, background=eps())
    nx, ny = size(P.c)    ### Dirichlet boundary conditions
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

    ### pressure and viscosity update
    for j = 1:ny
        for i = 1:nx
            dVxdx = (V.xc[i+1, j] - V.xc[i, j]) / dx 
            dVydy = (V.yc[i, j+1] - V.yc[i, j]) / dy

            P.c[i, j] = P₀.c[i, j] - γ * (dVxdx + dVydy)

            dVxdy_dVydx = 0.5*((V.xv[i+1, j+1] - V.xv[i+1, j]) / dy + (V.yv[i+1, j+1] - V.yv[i, j+1]) / dx)
            
            η.c[i, j] = 0.5 * B * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + background) ^ (0.5*q - 1)


        end
    end

    for j = 1:ny+1
        for i = 1:nx+1
            dVxdx = (V.xv[i+1, j] - V.xv[i, j]) / dx
            dVydy = (V.yv[i, j+1] - V.yv[i, j]) / dy

            P.v[i, j] = P₀.v[i, j] - γ * (dVxdx + dVydy)

            dVxdy = 1 < j < ny+1 ? 0.5 * (V.xc[i, j] - V.xc[i, j-1]) / dy : 0.  # gradient of wall parallel velocity is zero
            dVydx = 1 < i < nx+1 ? 0.5 * (V.yc[i, j] - V.yc[i-1, j]) / dx : 0.  # gradient of wall parallel velocity is zero

            η.v[i, j] = 0.5 * B * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + (dVxdy + dVydx)^2 + background) ^ (0.5*q - 1)

        end
    end

    ### residual in horizontal (x) direction
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


function construct_jacobian_with_boundary(n=5, seed=1234)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    B      = 0.1
    q    = 1.33
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    ρg   = deepcopy(P)
    η    = deepcopy(P)
    η̄    = deepcopy(P)
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=rand(rng, nx+1, ny  ), yc=rand(rng, nx  , ny+1),
            xv=rand(rng, nx+2, ny+1), yv=rand(rng, nx+1, ny+2))
    D    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))  # basis vector for directional derivative
    R    = deepcopy(D)  # Residuals
    Q    = deepcopy(D)  # row of Jacobian of compute_R wrt. V

    N_xc = (nx+1) * ny
    N_xv = (nx+2) * (ny+1)
    N_yc = nx * (ny+1)
    N_yv = (nx+1) * (ny+2)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = zeros(N, N) 

    E_xc = N_xc
    E_yc = E_xc + N_yc
    E_xv = E_yc + N_xv 

    col = 1
    for d = D
        for j = axes(d, 2)
            for i = axes(d, 1)
                # set one entry in search vector to 1
                d[i, j] = 1.0
                # compute the jacobian column by multiplying it with a "basis vector"
                autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q), Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, D), Const(ρg), Const(B), Const(q), Const(dx), Const(dy), Const(γ))
                # store result in jacobian
                J[1:E_xc, col]      .= reshape(Q.xc, N_xc)
                J[E_xc+1:E_yc, col] .= reshape(Q.yc, N_yc)
                J[E_yc+1:E_xv, col] .= reshape(Q.xv, N_xv)
                J[E_xv+1:end, col]  .= reshape(Q.yv, N_yv)
                # increase column count
                col += 1
                # reset search vector
                D.xc .= 0.
                D.yc .= 0.
                D.xv .= 0.
                D.yv .= 0.
            end
        end
    end

    return J
end

function construct_jacobian(n=5; seed=1)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    B      = 0.1
    q      = 1.33
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    ρg   = deepcopy(P)
    η    = deepcopy(P)
    η̄    = deepcopy(P)
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=rand(rng, nx+1, ny  ), yc=rand(rng, nx  , ny+1),
            xv=rand(rng, nx+2, ny+1), yv=rand(rng, nx+1, ny+2))
    D    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))  # basis vector for directional derivative
    R    = deepcopy(D)  # Residuals
    Q    = deepcopy(D)  # row of Jacobian of compute_R wrt. V
    #V = deepcopy(D)


    N_xc = (nx-1) * ny
    N_xv = (nx) * (ny+1)
    N_yc = nx * (ny-1)
    N_yv = (nx+1) * (ny)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = zeros(N, N) 

    E_xc = N_xc
    E_yc = E_xc + N_yc
    E_xv = E_yc + N_xv 

    col = 1
    for k = keys(D)
        for j = axes(D[k], 2)
            for i = axes(D[k], 1)
                # skip Dirichlet boundaries
                if (i == 1 || i == size(D[k], 1)) && (k ∈ [:xc, :xv]) continue end
                if (j == 1 || j == size(D[k], 2)) && (k ∈ [:yc, :yv]) continue end
                # set one entry in search vector to 1
                D[k][i, j] = 1.0
                # compute the jacobian column by multiplying it with a "basis vector"
                autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q), Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, D), Const(ρg), Const(B), Const(q), Const(dx), Const(dy), Const(γ))
                # store result in jacobian
                J[1:E_xc, col]      .= reshape(Q.xc[2:end-1, :], N_xc)
                J[E_xc+1:E_yc, col] .= reshape(Q.yc[:, 2:end-1], N_yc)
                J[E_yc+1:E_xv, col] .= reshape(Q.xv[2:end-1, :], N_xv)
                J[E_xv+1:end, col]  .= reshape(Q.yv[:, 2:end-1], N_yv)
                # increase column count
                col += 1
                # reset search vector
                D.xc .= 0.
                D.yc .= 0.
                D.xv .= 0.
                D.yv .= 0.
            end
        end
    end

    return J
end
