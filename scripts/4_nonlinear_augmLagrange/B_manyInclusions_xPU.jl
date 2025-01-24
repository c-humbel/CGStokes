using CairoMakie
using ColorSchemes
using LinearAlgebra
using Enzyme
using KernelAbstractions

function tplNorm(x::NamedTuple, p::Real=2)
    return norm(norm.(values(x), p), p)   
end


function tplDot(x::NamedTuple, y::NamedTuple, a::NamedTuple)
    s = 0.
    for k = keys(x)
        for I = eachindex(x[k])
            s += (x[k][I] * a[k][I] * y[k][I])
        end
    end
    return s
end


function tplDot(x::NamedTuple, y::NamedTuple, a::Real=1.)
    return a * sum(dot.(values(x), values(y)))
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::NamedTuple)
    for k = keys(dest)
        copyto!(dest[k], src[k])
        dest[k] .*= a[k]
    end
    return nothing
end


function tplSet!(dest::NamedTuple, src::NamedTuple, a::Real=1.)
    copyto!.(values(dest), values(src))
    tplScale!(dest, a)
    return nothing
end


function tplScale!(x::NamedTuple, a::Real)
    for k = keys(x)
        x[k] .*= a
    end
    return nothing
end


function tplAdd!(this::NamedTuple, other::NamedTuple)
    for k = keys(this)
        this[k] .+= other[k]
    end
    return nothing
end

function tplSub!(this::NamedTuple, other::NamedTuple)
    for k = keys(this)
        this[k] .-= other[k]
    end
    return nothing
end


# copied from 2_augmentedLagrange/D_ManyInclusions_Egrid.jl
function generate_inclusions(ninc, xs, ys, rng)
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    nx = length(xs)
    ny = length(ys)
    Lx = nx * dx
    Ly = ny * dy

    r_min = 2   * max(dx, dy)
    r_max = 0.1 * min(Lx, Ly)

    # generate random radii
    rs = r_min .+ (r_max - r_min) .* rand(rng, ninc)

    # generate random positions for non-overlapping circles
    xcs = zeros(ninc)
    ycs = zeros(ninc)
    i = 1
    while i <= ninc
        # generate guess
        xcs[i] = rand(rng, xs[end÷5:4end÷5])
        ycs[i] = rand(rng, ys[end÷5:4end÷5])
        # check that cicles are not overlapping with existing ones
        j = 1
        while j < i
            if (xcs[i] - xcs[j])^2 + (ycs[i] - ycs[j])^2 < (rs[i] + rs[j] + 2r_min)^2
                break
            end
            j += 1
        end     
        if j == i
            i += 1
        end 
    end
    return zip(xcs, ycs), rs
end

# copied from 2_augmentedLagrange/D_ManyInclusions_Egrid.jl
function initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly; seed=1234, ninc=5)
    rng = MersenneTwister(seed)

    # generate radius and location inclusions
    centers, radii = generate_inclusions(ninc, xc, yc, rng)

    # generate relative viscosity for inclusions
    η_ratios = fill(η_ratio, ninc)
    offsets = rand(rng, ninc-1) .* (η_ratio / 2)
    if η_ratio > 1
        η_ratios[2:end] .-= offsets
    elseif η_ratio < 1
        η_ratios[2:end] .+= offsets
    end
    shuffle!(rng, η_ratios)

    # area of inclusions
    As = [π*r^2 for r in radii]
    A_inc = sum(As)
    A_tot = Lx * Ly

    # matrix viscosity
    η_mat = η_avg * A_tot / (sum(As .* η_ratios) + A_tot - A_inc)

    # body force
    Δρg   = ρg_avg * A_tot / A_inc

    # set viscosity and body force values
    η_loc  = (c=Array(η.c), v=Array(η.v))
    ρg_loc = (c=Array(ρg.c), v=Array(ρg.v))
    η_loc.c  .= η_mat
    η_loc.v  .= η_mat
    ρg_loc.c .= 0.
    ρg_loc.v .= 0.
    for j = eachindex(yc)
        for i = eachindex(xc)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xc[i] - x)^2 + (yc[j] - y)^2 <= r^2
                    η_loc.c[i, j]  = η_rel * η_mat
                    ρg_loc.c[i, j] = Δρg
                    break
                end
            end
        end
    end

    for j = eachindex(yv)
        for i = eachindex(xv)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xv[i] - x)^2 + (yv[j] - y)^2 <= r^2
                    η_loc.v[i, j]  = η_rel * η_mat
                    ρg_loc.v[i, j] = Δρg
                    break
                end
            end
        end
    end
    
    copyto!(η.c, η_loc.c)
    copyto!(η.v, η_loc.v)
    copyto!(ρg.c, ρg_loc.c)
    copyto!(ρg.c, ρg_loc.v)

    return nothing
end


# dimensions for kernel launch: nx+1, ny+1
@kernel inbounds=true function compute_divV!(divV, V, iΔx, iΔy)
    i, j = @index(Global, NTuple)
    
    if i < size(V.xc, 1) && j < size(V.yc, 2)
        dVx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx
        dVy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy
        divV.c[i, j] = dVx + dVy
    end

    if i < size(V.xv, 1) && j < size(V.yv, 2)
        dVx = (V.xv[i+1, j] - V.xv[i, j]) * iΔx
        dVy = (V.yv[i, j+1] - V.yv[i, j]) * iΔy
        divV.v[i, j] = dVx + dVy
    end

    return nothing
end

# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function compute_R!(R, P, η, P₀, V, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ)
    i, j = @index(Global, NTuple)
    ### Dirichlet boundary conditions
    # wall normal velocities are zero
    if (i == 1 || i == size(V.xc, 1)) && j <= size(V.xc, 2)
        V.xc[i , j] = 0.
    end

    if (i == 1 || i == size(V.xv, 1)) && j <= size(V.xv, 2)
        V.xv[i , j] = 0.
    end

    if i <= size(V.yc, 1) && (j == 1 || j == size(V.yc, 2))
        V.yc[i, j] = 0.
    end

    if i <= size(V.yv, 1) && (j == 1 || j == size(V.yv, 2))
        V.yv[i, j] = 0.
    end

    ### pressure and viscosity update
    if i < size(V.xc, 1) && j < size(V.yc, 2)
        dVxdx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy

        P.c[i, j] = P₀.c[i, j] - γ * (dVxdx + dVydy)

        dVxdy_dVydx = 0.5*((V.xv[i+1, j+1] - V.xv[i+1, j]) * iΔy + (V.yv[i+1, j+1] - V.yv[i, j+1]) * iΔx )
        
        η.c[i, j] = 0.5 * B.c[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)
    end

    if i < size(V.xv, 1) && j < size(V.yv, 2)
        dVxdx = (V.xv[i+1, j] - V.xv[i, j]) * iΔx
        dVydy = (V.yv[i, j+1] - V.yv[i, j]) * iΔy

        P.v[i, j] = P₀.v[i, j] - γ * (dVxdx + dVydy)

        dVxdy = 1 < j < size(V.yv, 2)-1 ? 0.5 * (V.xc[i, j] - V.xc[i, j-1]) * iΔy : 0.  # gradient of wall parallel velocity is zero
        dVydx = 1 < i < size(V.xv, 1)-1 ? 0.5 * (V.yc[i, j] - V.yc[i-1, j]) * iΔx : 0.  # gradient of wall parallel velocity is zero

        η.v[i, j] = 0.5 * B.v[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + (dVxdy + dVydx)^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)

    end

    ### residual in horizontal (x) directio
    ## including Neumann BC on at top and bottom boundary
    ## for velocities associated with cell centers (V.xc)
    if 1 < i < size(η.c, 1) && j <= size(η.c, 2)
        # all values in y direction
        # inner values in x direction
        # stress at horizontally adjacent cell centers
        τxx_r = 2 * η.c[i  , j] * (V.xc[i+1, j] - V.xc[i  , j]) * iΔx
        τxx_l = 2 * η.c[i-1, j] * (V.xc[i  , j] - V.xc[i-1, j]) * iΔx

        # stress at vertically adjacent cell corners
        if j > 1
            τxy_b = η.v[i, j  ] * ((V.xc[i, j  ] - V.xc[i, j-1]) * iΔy + (V.yc[i, j  ] - V.yc[i-1, j  ]) * iΔx)
        else
            τxy_b = 0.  # zero stress at the bottom boundary
        end

        if j < size(η.c, 2)
            τxy_t = η.v[i, j+1] * ((V.xc[i, j+1] - V.xc[i, j  ]) * iΔy + (V.yc[i, j+1] - V.yc[i-1, j+1]) * iΔx)
        else
            τxy_t = 0.  # zero stress at the top boundary
        end

        # residual in x direction on the interface
        R.xc[i, j] = ( (τxx_r - τxx_l) * iΔx
                     + (τxy_t - τxy_b) * iΔy
                     - (P.c[i, j] - P.c[i-1, j]) * iΔx)
    end
    ## for velocities associated with cell corners (V.xv)
    if 1 < i < size(η.v, 1) && j <= size(η.v, 2)
        # all values in y direction
        # inner values in x direction
        τxx_r = 2 * η.v[i  , j] * (V.xv[i+1, j] - V.xv[i  , j]) * iΔx
        τxx_l = 2 * η.v[i-1, j] * (V.xv[i  , j] - V.xv[i-1, j]) * iΔx

        if j > 1
            τxy_b = η.c[i-1, j-1] * ((V.xv[i, j  ] - V.xv[i, j-1]) * iΔy + (V.yv[i, j  ] - V.yv[i-1, j  ]) * iΔx)
        else
            τxy_b = 0.  # zero stress at the bottom boundary
        end

        if j < size(η.v, 2)
            τxy_t = η.c[i-1, j  ] * ((V.xv[i, j+1] - V.xv[i, j  ])* iΔy + (V.yv[i, j+1] - V.yv[i-1, j+1]) * iΔx)
        else
            τxy_t = 0.  # zero stress at the top boundary
        end

        R.xv[i, j] = ( (τxx_r - τxx_l) * iΔx
                     + (τxy_t - τxy_b) * iΔy
                     - (P.v[i, j] - P.v[i-1, j]) * iΔx)
    end

    ### residual in vertical (y) direction
    ### including Neumann BC at left and right boundary
    ## for velocities associated with cell centers (V.yc)
    if i <= size(η.c, 1) && 1 < j < size(η.c, 2)
        # inner values in y direction
        # all values in x direction
        τyy_t = 2 * η.c[i, j  ] * (V.yc[i, j+1] - V.yc[i, j  ]) * iΔy
        τyy_b = 2 * η.c[i, j-1] * (V.yc[i, j  ] - V.yc[i, j-1]) * iΔy

        if i > 1
            τxy_l = η.v[i  , j] * ((V.xc[i  , j] - V.xc[i  , j-1]) * iΔy + (V.yc[i  , j] - V.yc[i-1, j]) * iΔx)
        else
            τxy_l = 0.  # zero stress at the left boundary
        end

        if i < size(η.c, 1)
            τxy_r = η.v[i+1, j] * ((V.xc[i+1, j] - V.xc[i+1, j-1]) * iΔy + (V.yc[i+1, j] - V.yc[i  , j]) * iΔx)
        else
            τxy_r = 0.  # zero stress at the right boundary
        end
        
        R.yc[i, j] = ( (τyy_t - τyy_b)  * iΔy
                     + (τxy_r - τxy_l) * iΔx
                     - ( P.c[i, j] -  P.c[i, j-1])  * iΔy
                     - (ρg.c[i, j] + ρg.c[i, j-1]) * 0.5)
    end
    ## for velocities associated with cell corners (V.yv)
    if i <= size(η.v, 1) && 1 < j < size(η.v, 2)
        τyy_t = 2 * η.v[i, j  ] * (V.yv[i, j+1] - V.yv[i, j  ]) * iΔy
        τyy_b = 2 * η.v[i, j-1] * (V.yv[i, j  ] - V.yv[i, j-1]) * iΔy

        if i > 1
            τxy_l = η.c[i-1, j-1] * ((V.xv[i  , j] - V.xv[i  , j-1]) * iΔy + (V.yv[i  , j] - V.yv[i-1, j]) * iΔx)
        else
            τxy_l = 0.  # zero stress at the left boundary
        end

        if i < size(η.v, 1)
            τxy_r = η.c[i  , j-1] * ((V.xv[i+1, j] - V.xv[i+1, j-1]) * iΔy + (V.yv[i+1, j] - V.yv[i  , j]) * iΔx)
        else
            τxy_r = 0.  # zero stress at the right boundary
        end

        R.yv[i, j] = ( (τyy_t - τyy_b) * iΔy
                     + (τxy_r - τxy_l) * iΔx
                     - ( P.v[i, j] -  P.v[i, j-1]) * iΔy
                     + (ρg.v[i, j] + ρg.v[i, j-1]) * 0.5)
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
    return nothing
end

# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function update_D!(D, R, invM, β)
    i, j = @index(Global, NTuple)
    if 1 < i < size(D.xc, 1) && j <= size(D.xc, 2)
        D.xc[i, j] = invM.xc[i, j] * R.xc[i, j] + β * D.xc[i, j]
    end

    if i <= size(D.yc, 1) && 1 < y < size(D.yc, 2)
        D.yc[i, j] = invM.yc[i, j] * R.yc[i, j] + β * D.yc[i, j]
    end

    if 1 < i < size(D.xv, 1) && y < size(D.xv, 2)
        D.xv[i, j] = invM.xv[i, j] * R.xv[i, j] + β * D.xv[i, j]
    end

    if i < y < size(D.yv, 1) && 1 < j < size(D.yv, 2)
        D.yv[i, j] = invM.yv[i, j] * R.yv[i, j] + β * D.yv[i, j]
    end

    return nothing
end

# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function update_V!(V, D, α)
    i, j = @index(Global, NTuple)
    if 1 < i < size(V.xc, 1)&& j <= :size(V.xc, 2)
        V.xc[i, j] += α * D.xc[i, j]
    end

    if i <= size(V.yc, 1) && 1 < j < size(V.yc, 2)
        V.yc[i, j] += α * D.yc[i, j]
    end

    if 1 < i < size(V.xv, 1) && j <= size(V.xv, 2)
        V.xv[i, j] += α * D.xv[i, j]
    end

    if i <= size(V.yv, 1) && 1 < j <size(V.yv, 2)
        V.yv[i, j] += α * D.yv[i, j]
    end
    return nothing
end


@kernel inbounds=true function initialise_invM(invM, η, iΔx, iΔy, γ)
    ## inner points
    # x direction, cell centers
    if 1 < i < size(invM.xc, 1) && 1 < j < size(invM.xc, 2)
        mij = ( 2iΔx^2 * (η.c[i-1, j] + η.c[i, j  ])
               + iΔy^2 * (η.v[i  , j] + η.v[i, j+1])
               + 2γ * iΔx^2)
        invM.xc[i, j] = inv(mij)
    end

    # y direction, cell centers
    if 1 < i < size(invM.yc, 1) && 1 < j < size(invM.yc, 2)
        mij = ( 2iΔy^2 * (η.c[i, j-1] + η.c[i  , j])
               + iΔx^2 * (η.v[i, j  ] + η.v[i+1, j])
               + 2γ * iΔy^2)
        invM.yc[i, j] = inv(mij)
    end

    # x direction, vertices
    if 1 < i < size(invM.xv, 1) && 1 < j < size(invM.xv, 2)
        mij = ( 2iΔx^2 * (η.v[i-1, j  ] + η.v[i  , j])
               + iΔy^2 * (η.c[i-1, j-1] + η.c[i-1, j])
               + 2γ * iΔx^2)
        invM.xv[i, j] = inv(mij)
    end

    # y direction, vertices
    if 1 < i < size(invM.yv, 1) && 1 < j < size(invM.yv, 2)
        mij = ( 2iΔy^2 * (η.v[i  , j-1] + η.v[i, j  ])
                + iΔx^2 * (η.c[i-1, j-1] + η.c[i, j-1])
                + 2γ *iΔy^2)
        invM.yv[i, j] = inv(mij)
    end

    ## Neumann boundary points
    # x direction, cell centers
    if 1 < i < size(invM.xc, 1)
        if j == 1
            invM.xc[i, j] = inv(2iΔx^2 * (η.c[i-1, j] + η.c[i, j])
                               + iΔy^2 * (η.v[i, j+1])
                               + 2γ * iΔx^2)
        elseif j == size(invM.xc, 2)
            invM.xc[i, j] = inv(2iΔx^2 * (η.c[i-1, j] + η.c[i, j])
                               + iΔy^2 * (η.v[i  , j])
                               + 2γ * iΔx^2)
        end
    end
    # y direction, cell centers
    if 1 < j < size(invM.yc, 2)
        if i == 1
            invM.yc[i, j] = inv(2iΔy^2 * (η.c[i, j-1] + η.c[i, j])
                               + iΔx^2 * (η.v[i+1, j])
                               + 2γ * iΔy^2)
        end
        if i == size(invM.yc, 1)
            invM.yc[i, j] = inv(2iΔy^2 * (η.c[i, j-1] + η.c[i, j])
                               + iΔx^2 * (η.v[i, j  ])
                               + 2γ * iΔy^2)
        end
    end
    # x direction, vertices
    if 1 < i < size(invM.xv, 1)
        if j == 1
            invM.xv[i, j] = inv(2iΔx^2 * (η.v[i-1, j] + η.v[i, j])
                               + iΔy^2 * (η.c[i-1, j])
                               + 2γ * iΔx^2)
        end
        if j == size(invM.xv, 2)
            invM.xv[i, j] = inv(2iΔx^2 * (η.v[i-1, j] + η.v[i, j])
                               + iΔy^2 * (η.c[i-1, j-1])
                               + 2γ * iΔx^2)
        end
    end
    # y direction, vertices
    if 1 < j < size(invM.yv, 2)
        if i == 1
            invM.yv[i, j] = inv(2iΔy^2 * (η.v[i, j-1] + η.v[i, j])
                               + iΔx^2 * (η.c[i, j-1])
                               + 2γ * iΔy^2)
        end
        if i == size(invM.yv, 1)
            invM.yv[i, j] = inv(2iΔy^2 * (η.v[i  , j-1] + η.v[i, j])
                               + iΔx^2 * (η.c[i-1, j-1])
                               + 2γ * iΔy^2)
        end
    end

    ## Dirichlet boundary points, leave zero

    return nothing
    
end


function nonlinear_inclusion(;n=127, η_ratio=0.1, niter=10000, γ_factor=1.,
                            ϵ_cg=1e-3, ϵ_ph=1e-6, ϵ_newton=1e-3, verbose=false)
    L_ref =  1. # reference length 
    ρg_avg = 1. # average density

    # physical parameters would be: n == 3, A = (24 * 1e-25) in Glen's law, see Cuffrey and Paterson (2006), table 3.3
    # and q = 1. + 1/n, η_avg = (24 * 1e-25) ^ (-1/n), see Schoof (2006)
    η_avg = 1. 
    q = 1. + 1/3  

    Lx = Ly = L_ref
    nx = ny = n
    ϵ̇_bg = eps()

    dx, dy = Lx / nx, Ly / ny
    xc = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    yc = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)
    xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
    yv = LinRange(-0.5Ly, 0.5Ly, ny+1)


    # field initialisation
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)
    ρg   = deepcopy(P)
    B    = deepcopy(P)
    η    = deepcopy(P)  # viscosity
    η̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=zeros(nx+1, ny), yc=zeros(nx, ny+1), xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))
    dV   = deepcopy(V)  # velocity updates in Newton iteration
    V̄    = deepcopy(V)  # memory needed for auto-differentiation
    D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
    R    = deepcopy(V)  # nonlinear Residual
    K    = deepcopy(V)  # Residuals in CG
    Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by some vector (used for autodiff)
    invM = deepcopy(V)  # preconditioner, cells correspoinding to Dirichlet BC are zero
    
    initialise_η_ρ!(η, ρg, η_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly)


    # Coefficient of augmented Lagrangian
    γ = γ_factor * tplNorm(B, Inf)

    # residual norms for monitoring convergence
    δ = Inf # CG
    χ = Inf # Newton
    ω = Inf # Pressure

    δ_ref = tplNorm(ρg, Inf) # is this correct ?
    ω_ref = ρg_avg * Lx / η_avg

    # visualisation
    fig = Figure(size=(600,400))
    axs = (Eta=Axis(fig[1,1][1,1], aspect=1), P=Axis(fig[1,2][1,1], aspect=1),
           Vx=Axis(fig[2,1][1,1], aspect=1), Vy=Axis(fig[2,2][1,1], aspect=1))
    plt = (Eta=heatmap!(axs.Eta, η.c, colormap=ColorSchemes.viridis),
           P=heatmap!(axs.P, P.c, colormap=ColorSchemes.viridis),
           Vx=heatmap!(axs.Vx, V.xc, colormap=ColorSchemes.viridis),
           Vy=heatmap!(axs.Vy, V.yc, colormap=ColorSchemes.viridis))
    cbar= (Eta=Colorbar(fig[1, 1][1, 2], plt.Eta),
           P=Colorbar(fig[1, 2][1, 2], plt.P),
           Vx=Colorbar(fig[2, 1][1, 2], plt.Vx),
           Vy=Colorbar(fig[2, 2][1, 2], plt.Vy))

    display(fig)

    # Powell Hestenes
    it = 0
    while it < niter && ω > ϵ_ph
        verbose && println("Iteration ", it_P)
        tplSet!(P₀, P)

        compute_R!(R, P,  η, P₀, V, ρg, B, q, ϵ̇_bg, dx, dy, γ)

        χ = tplNorm(R, Inf) / δ_ref

        # Newton iteration
        while it < niter && χ > ϵ_newton
            # initialise preconditioner
            initialise_invM(invM, η, dx, dy, γ)

            # iteration zero
            # compute residual for CG, K = R - DR * dV
            tplSet!(V̄, dV)
            autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
                     Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                     Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(dx), Const(dy), Const(γ))
            tplSet!(K, R)
            tplAdd!(K, Q)

            tplSet!(D, K, invM)
            μ = tplDot(K, D)
            δ = tplNorm(K, Inf) / δ_ref
            # start iteration
            while it <= niter && δ > ϵ_cg
                # compute α
                tplSet!(V̄, D)
                autodiff(Forward, compute_R!, DuplicatedNoNeed(K, Q),
                     Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                     Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(dx), Const(dy), Const(γ))

                α = μ / tplDot(D, Q, -1.)

                update_V!(dV, D, α)

                # recompute residual
                tplSet!(V̄, dV)
                autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
                         Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀), Duplicated(V, V̄),
                         Const(ρg), Const(B), Const(q), Const(ϵ̇_bg), Const(dx), Const(dy), Const(γ))
                tplSet!(K, R)
                tplAdd!(K, Q)


                μ_new = tplDot(K, K, invM)
                β = μ_new / μ
                μ = μ_new
                update_D!(D, K, invM, β)

                # compute residual norm
                δ = tplNorm(K, Inf) / δ_ref # correct scaling?
                it += 1

                if it % 10 == 0 println("CG residual = ", δ) end
            end
            tplAdd!(V, dV)

            # update plot
            plt.Eta[3][] .= log10.(η.c)
            plt.P[3][]   .= P.c
            plt.Vx[3][]  .= V.xc
            plt.Vy[3][]  .= V.yc
            plt.Eta.colorrange[]= (min(-1,log10(minimum(η.c))), max(1,log10(maximum(η.c))))
            plt.P.colorrange[]  = (min(-1e-10,minimum(P.c)), max(1e-10, maximum(P.c)))
            plt.Vx.colorrange[] = (min(-1e-10,minimum(V.xc)), max(1e-10,maximum(V.xc)))
            plt.Vy.colorrange[] = (min(-1e-10,minimum(V.yc)), max(1e-10,maximum(V.yc)))


            display(fig)

            compute_R!(R, P,  η, P₀, V, ρg, B, q, ϵ̇_bg, dx, dy, γ)
            χ = tplNorm(R, Inf) / δ_ref # correct scaling?
            println("Newton residual = ", χ, "; total iteration count: ", it)
        end    
        compute_divV!(divV, V, dx, dy)
        ω = tplNorm(divV, Inf) / ω_ref # correct scaling?
        println("Pressure residual = ", ω, ", Newton residual = ", χ, ", CG residual = ", δ)
    end

    return it, P, V, R, η
end


outfields = nonlinear_inclusion(n=64, niter=5000, ϵ_ph=1e-3, ϵ_cg=1e-3, ϵ_newton=0.5);

