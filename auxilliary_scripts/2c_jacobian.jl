using Enzyme
using LinearAlgebra
using SparseArrays

# copy of the residual computation in 2_augmentedLagrangian2D.jl

# record changes that have been made to the original code
# 1. resize residuals to the same shape as the velocity field, such that the Jacobian is square
#    (now we could iterate from 2 to end-1 in the update functions, but that would mean changing all accesses to other fields)
# 2. add boundary conditions to the residuals
# 3. Insead of having residuals from BC, apply them directly to the velocity field (supersedes point 2)
# 4. split Neumann and Dirchlet BC: apply Dirichlet, and include Neumann in the residual (supersedes point 2,3)
function compute_R!(R, P, P_old, V, ρg, η, dx, dy, γ)

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
    ##  including Neumann BC on Vx at top and bottom boundary
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
    return nothing
end

function construct_jacobian(n=5)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    η      =  rand(nx, ny)
    ρg     = zeros(nx, ny)
    P      = zeros(nx, ny)
    P_old  = zeros(nx, ny)
    P_tmp  = zeros(nx, ny)
    V      = (x = rand(nx+1, ny  ), y = rand(nx  , ny+1))
    e      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) # basis vectors
    R      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) 
    colJ   = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) # variable to store columns of the jacobian

    nxe = (nx-1) * ny
    nye = nx * (ny-1)
    J   = zeros(nxe + nye, nxe + nye) 

    col = 1
    for j = 1:ny
        for i = 2:nx
            # set one entry in search vector to 1
            e.x[i, j] = 1.0
            # compute the jacobian column by multiplying it with a "basis vector"
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, P_tmp), Const(P_old), Duplicated(V, e), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
            # store result in jacobian
            # remove cells affected by Dirichlet BC ("ghost cells")
            J[1:nxe, col]     .= reshape(colJ.x[2:end-1, :], nxe)
            J[nxe+1:end, col] .= reshape(colJ.y[:, 2:end-1], nye)
            # increase column count
            col += 1
            # reset search vector
            e.x[i, j] = 0.0
        end
    end

    for j = 2:ny
        for i = 1:nx
            e.y[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, P_tmp), Const(P_old), Duplicated(V, e), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
            J[1:nxe, col] .= reshape(colJ.x[2:end-1, :], nxe)
            J[nxe+1:end, col] .= reshape(colJ.y[:, 2:end-1], nye)
            col += 1
            e.y[i, j] = 0.0
        end
    end

    return J
end

J = construct_jacobian();

spJ = sparse(J)

@assert issparse(spJ)
@assert isposdef(spJ)
 
