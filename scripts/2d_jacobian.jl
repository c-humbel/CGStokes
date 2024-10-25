using Enzyme
using LinearAlgebra
using SparseArrays

# copy of the residual computation in 2_augmentedLagrangian2D.jl

# record changes that have been made to the original code
# 1. resize residuals to the same shape as the velocity field, such that the Jacobian is square
#    (now we could iterate from 2 to end-1 in the update functions, but that would mean changing all accesses to other fields)
# 2. add boundary conditions to the residuals
function compute_R!(R, P, P_old, V, ρg, η, dx, dy, γ)
    nx, ny = size(P)

    # pressure update
    for j = 1:ny
        for i = 1:nx
            P[i, j] = P_old[i, j] - γ * ((V.x[i+1, j] - V.x[i, j]) / dx + (V.y[i, j+1] - V.y[i, j]) / dy)
        end
    end

    # compute the residual at cell interfaces
    # in horizontal (x) direction
    for j = 1:ny-2
        for i = 1:nx-1
            # stress at horizontally adjacent cell centers
            τxx_r = 2 * η[i+1, j+1] * (V.x[i+2, j+1] - V.x[i+1, j+1]) / dx
            τxx_l = 2 * η[i  , j+1] * (V.x[i+1, j+1] - V.x[i  , j+1]) / dx
            # viscosity at vertically adjacent cell corners
            η_t   = 0.25 * (η[i, j+1] + η[i+1, j+1] + η[i, j+2] + η[i+1, j+2])
            η_b   = 0.25 * (η[i, j  ] + η[i+1, j  ] + η[i, j+1] + η[i+1, j+1])
            # stress at same cell corners
            τxy_t = η_t * ((V.x[i+1, j+2] - V.x[i+1, j+1]) / dy
                         + (V.y[i+1, j+2] - V.y[i  , j+2]) / dx)
            τxy_b = η_b * ((V.x[i+1, j+1] - V.x[i+1, j  ]) / dy
                         + (V.y[i+1, j+1] - V.y[i  , j+1]) / dx)
            # residual in x direction on the interface
            R.x[i+1, j+1]  = (- (τxx_r - τxx_l) / dx
                              - (τxy_t - τxy_b) / dy
                              + (P[i+1, j+1] - P[i, j+1]) / dx)
        end
    end

    # residual in y direction
    for j = 1:ny-1
        for i = 1:nx-2
            τyy_t = 2 * η[i+1, j+1] * (V.y[i+1, j+2] - V.y[i+1, j+1]) / dy
            τyy_b = 2 * η[i+1, j  ] * (V.y[i+1, j+1] - V.y[i+1, j  ]) / dy

            η_r   = 0.25 * (η[i+1, j  ] + η[i+2, j  ] + η[i+1, j+1] + η[i+2, j+1])
            η_l   = 0.25 * (η[i  , j  ] + η[i+1, j  ] + η[i  , j+1] + η[i+1, j+1])

            τxy_r = η_r * ((V.x[i+2, j+1] - V.x[i+2, j  ]) / dy
                         + (V.y[i+2, j+1] - V.y[i+1, j+1]) / dx)
            τxy_l = η_l * ((V.x[i+1, j+1] - V.x[i+1, j  ]) / dy
                         + (V.y[i+1, j+1] - V.y[i  , j+1]) / dx)
            
            R.y[i+1, j+1] = ( - (τyy_t - τyy_b) / dy
                              - (τxy_r - τxy_l) / dx
                              + (P[i+1, j+1] - P[i+1, j]) / dy
                              - (ρg[i+1, j+1] + ρg[i+1, j]) * 0.5)
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
        # multiply by viscosity / dy to get symmetric matrix entries (has to match with inner points)
        η_t = 0.25 * (η[i-1, 1  ] + η[i, 1  ] + η[i-1, 2    ] + η[i, 2    ])
        η_b = 0.25 * (η[i-1, end] + η[i, end] + η[i-1, end-1] + η[i, end-1])
        R.x[i, 1  ] = -η_t * (V.x[i, 2    ] - V.x[i, 1  ]) / dy^2
        R.x[i, end] = -η_b * (V.x[i, end-1] - V.x[i, end]) / dy^2
    end

    for j = 2:ny
        η_r   = 0.25 * (η[end, j-1] + η[end-1, j-1] + η[end, j] + η[end-1, j])
        η_l   = 0.25 * (η[1  , j-1] + η[2    , j-1] + η[1  , j] + η[2    , j])
        R.y[1  , j] = -η_l * (V.y[2    , j] - V.y[1  , j]) / dx^2
        R.y[end, j] = -η_r * (V.y[end-1, j] - V.y[end, j]) / dx^2
    end
    return nothing
end


# simplified version of the residual computation to check if the jacobian is assembled correctly
function simple_R(R, P, P_old, V, ρg, η, dx, dy, γ)
    nx, ny = size(P)
    for j = 0:ny-1
        for i = 1:nx-1
            R.x[i+1, j+1]  = - (V.x[i+2, j+1] - V.x[i, j+1]) / 2dx
        end
    end

    for j = axes(R.x, 2)
        R.x[1, j]   = - (V.x[2  , j] - V.x[1     , j]) / dx
        R.x[end, j] = - (V.x[end, j] - V.x[end-1, j]) / dx
    end

    # residual in y direction
    for j = 1:ny-1
        for i = 0:nx-1
            R.y[i+1, j+1]  = - (V.y[i+1, j+2] - V.y[i+1, j]) / 2dy
        end
    end

    for i = axes(R.y, 1)
        R.y[i, 1]   = - (V.y[i, 2  ] - V.y[i, 1    ]) / dy
        R.y[i, end] = - (V.y[i, end] - V.y[i, end-1]) / dy
    end
    return nothing
end


function construct_jacobian(n=5)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 1.0
    η      =  ones(nx, ny)
    ρg     = zeros(nx, ny)
    P      = zeros(nx, ny)
    P_old  = zeros(nx, ny)
    P_tmp  = zeros(nx, ny)
    V      = (x = rand(nx+1, ny  ), y = rand(nx  , ny+1))
    e      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) # basis vectors
    R      = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) 
    colJ   = (x =zeros(nx+1, ny  ), y =zeros(nx  , ny+1)) # variable to store columns of the jacobian

    nrows = length(colJ.x) + length(colJ.y)
    ncols = length(V.x) + length(V.y)
    J     = zeros(nrows, ncols) 

    col = 1
    for j = axes(e.x, 2)
        for i = axes(e.x, 1)
            # set one entry in search vector to 1
            e.x[i, j] = 1.0
            # compute the jacobian column by multiplying it with a "basis vector"
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, P_tmp), Const(P_old), Duplicated(V, e), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
            # store result in jacobian
            J[1:length(colJ.x), col] .= reshape(colJ.x, length(colJ.x))
            J[length(colJ.x)+1:end, col] .= reshape(colJ.y, length(colJ.y))
            # increase column count
            col += 1
            # reset search vector
            e.x[i, j] = 0.0
        end
    end

    for j = axes(e.y, 2)
        for i = axes(e.y, 1)
            e.y[i, j] = 1.0
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, colJ), Duplicated(P, P_tmp), Const(P_old), Duplicated(V, e), Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
            J[1:length(colJ.x), col] .= reshape(colJ.x, length(colJ.x))
            J[length(colJ.x)+1:end, col] .= reshape(colJ.y, length(colJ.y))
            col += 1
            e.y[i, j] = 0.0
        end
    end

    return J
end

J = construct_jacobian(8);

sparse(J)
d = sparse(J - J')