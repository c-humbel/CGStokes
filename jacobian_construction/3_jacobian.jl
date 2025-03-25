using Enzyme
using LinearAlgebra
using SparseArrays
using Random

# copy of the residual computation in 3_CGlinearStokes2D.jl

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
            R.p[i, j] = -(dVx + dVy)
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
                        - (ρg[i, j] + ρg[i, j-1]) * 0.5)
        end
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
    return nothing
end

function construct_jacobian_with_boundary(n=5; seed=1)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    η      = rand(rng, nx, ny)
    ρg     = zeros(nx, ny)
    P      = rand(nx, ny)
    
    V      = (x = rand(nx+1, ny), y = rand(nx  , ny+1))
    ev     = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1))                     # basis vectors for velocity part
    ep     = zeros(nx, ny)                                                  # basis vectors for pressure part
    R      = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1), p=zeros(nx, ny)) 
    colJ   = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1), p=zeros(nx, ny))    # variable to store columns of the jacobian

    nxe = (nx+1) * ny
    nye = nx * (ny+1)
    npe = nx * ny
    E_x = nxe
    E_y = nxe + nye
    E_p = nxe + nye + npe
    J   = zeros(E_p,E_p) 


    col = 1
    for j = 1:ny
        for i = 1:nx+1
            # set one entry in search vector to 1
            ev.x[i, j] = 1.0
            # compute the jacobian column by multiplying it with a "basis vector"
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            # store result in jacobian
            J[1:E_x, col]     .= reshape(colJ.x, nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y, nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            # increase column count
            col += 1
            # reset search vector
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0

        end
    end

    for j = 1:ny+1
        for i = 1:nx
            ev.y[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            J[1:E_x, col]     .= reshape(colJ.x, nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y, nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            col += 1
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0
        end
    end

    for j = 1:ny
        for i = 1:nx
            ep[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            J[1:E_x, col]     .= reshape(colJ.x, nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y, nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            col += 1
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0
        end
    end
    return J
end

function construct_jacobian(n=5; seed=1)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    η      = rand(rng, nx, ny)
    ρg     = zeros(nx, ny)
    P      = rand(nx, ny)
    
    V      = (x = rand(nx+1, ny), y = rand(nx  , ny+1))
    ev     = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1))                     # basis vectors for velocity part
    ep     = zeros(nx, ny)                                                  # basis vectors for pressure part
    R      = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1), p=zeros(nx, ny)) 
    colJ   = (x =zeros(nx+1, ny), y =zeros(nx  , ny+1), p=zeros(nx, ny))    # variable to store columns of the jacobian

    nxe = (nx-1) * ny
    nye = nx * (ny-1)
    npe = nx * ny
    E_x = nxe
    E_y = nxe + nye
    E_p = nxe + nye + npe
    J   = zeros(E_p,E_p) 


    col = 1
    for j = 1:ny
        for i = 2:nx
            # set one entry in search vector to 1
            ev.x[i, j] = 1.0
            # compute the jacobian column by multiplying it with a "basis vector"
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            # store result in jacobian
            J[1:E_x, col]     .= reshape(colJ.x[2:nx, :], nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y[:, 2:ny], nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            # increase column count
            col += 1
            # reset search vector
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0

        end
    end

    for j = 2:ny
        for i = 1:nx
            ev.y[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            J[1:E_x, col]     .= reshape(colJ.x[2:nx, :], nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y[:, 2:ny], nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            col += 1
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0
        end
    end

    for j = 1:ny
        for i = 1:nx
            ep[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, ep), Duplicated(V, ev), Const(ρg), Const(η), Const(dx), Const(dy))
            J[1:E_x, col]     .= reshape(colJ.x[2:nx, :], nxe)
            J[E_x+1:E_y, col] .= reshape(colJ.y[:, 2:ny], nye)
            J[E_y+1:E_p, col] .= reshape(colJ.p, npe)
            col += 1
            ev.x .= 0.0
            ev.y .= 0.0
            ep .= 0.0
        end
    end
    return J
end


#TODO: construct diagonal preconditioner for full system
#= function construct_M(n=5, seed=1)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    η      = rand(rng, nx, ny)
    Minv = (x = zeros(nx+1, ny), y = zeros(nx, ny+1))

    ## inner points
    # x direction
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
    return Minv
    
end =#

n = 5

## check that the jacobian with BC is symmetric
J = construct_jacobian_with_boundary(n);

@assert issymmetric(J)

## check that the jacobian without BC is symmetric
Jin = construct_jacobian(n);
@assert issymmetric(Jin)
# but it is not positive definite
@assert !isposdef(Jin)

 
# ## check that the preconditioner is correct
# M = construct_M(n);
# m = vcat(reshape(M.x, length(M.x)), reshape(M.y, length(M.y)));

# # reference values of the preconditioner
# mexact = diag(-J);
# minv_exact = zeros(length(mexact));
# minv_exact[mexact .!= 0] = inv.(mexact[mexact .!= 0]);

# @assert all(m .≈ minv_exact)
