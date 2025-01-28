using CairoMakie
using ColorSchemes
using Enzyme
using Random

include("../../src/tuple_manip.jl")

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
    η.c  .= η_mat
    η.v  .= η_mat
    ρg.c .= 0.
    ρg.v .= 0.
    for j = eachindex(yc)
        for i = eachindex(xc)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xc[i] - x)^2 + (yc[j] - y)^2 <= r^2
                    η.c[i, j]  = η_rel * η_mat
                    ρg.c[i, j] = Δρg
                    break
                end
            end
        end
    end

    for j = eachindex(yv)
        for i = eachindex(xv)
            for ((x, y), r, η_rel) ∈ zip(centers, radii, η_ratios)
                if (xv[i] - x)^2 + (yv[j] - y)^2 <= r^2
                    η.v[i, j]  = η_rel * η_mat
                    ρg.v[i, j] = Δρg
                    break
                end
            end
        end
    end


    return nothing
end

function compute_divV!(divV, V, dx, dy)
    nx, ny = size(divV.c)
    for j = 1:ny
        for i = 1:nx
            dVx = (V.xc[i+1, j] - V.xc[i, j]) / dx
            dVy = (V.yc[i, j+1] - V.yc[i, j]) / dy
            divV.c[i, j] = dVx + dVy
        end
    end

    for j = 1:ny+1
        for i = 1:nx+1
            dVx = (V.xv[i+1, j] - V.xv[i, j]) / dx
            dVy = (V.yv[i, j+1] - V.yv[i, j]) / dy
            divV.v[i, j] = dVx + dVy
        end
    end

    return nothing
end

function compute_R!(R, P, η, P₀, V, ρg, B, q, ϵ̇_bg, dx, dy, γ)
    nx, ny = size(P.c)

    ### Dirichlet boundary conditions
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
            
            η.c[i, j] = 0.5 * B.c[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)


        end
    end

    for j = 1:ny+1
        for i = 1:nx+1
            dVxdx = (V.xv[i+1, j] - V.xv[i, j]) / dx
            dVydy = (V.yv[i, j+1] - V.yv[i, j]) / dy

            P.v[i, j] = P₀.v[i, j] - γ * (dVxdx + dVydy)

            dVxdy = 1 < j < ny+1 ? 0.5 * (V.xc[i, j] - V.xc[i, j-1]) / dy : 0.  # gradient of wall parallel velocity is zero
            dVydx = 1 < i < nx+1 ? 0.5 * (V.yc[i, j] - V.yc[i-1, j]) / dx : 0.  # gradient of wall parallel velocity is zero

            η.v[i, j] = 0.5 * B.v[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + (dVxdy + dVydx)^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)

        end
    end

    ### residual in horizontal (x) directio
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


function update_D!(D, R, invM, β)
    for j = 1:size(D.xc, 2)
        for i = 2:size(D.xc, 1)-1
            D.xc[i, j] = invM.xc[i, j] * R.xc[i, j] + β * D.xc[i, j]
        end
    end

    for j = 2:size(D.yc, 2)-1
        for i = 1:size(D.yc, 1)
            D.yc[i, j] = invM.yc[i, j] * R.yc[i, j] + β * D.yc[i, j]
        end
    end

    for j = 1:size(D.xv, 2)
        for i = 2:size(D.xv, 1)-1
            D.xv[i, j] = invM.xv[i, j] * R.xv[i, j] + β * D.xv[i, j]
        end
    end

    for j = 2:size(D.yv, 2)-1
        for i = 1:size(D.yv, 1)
            D.yv[i, j] = invM.yv[i, j] * R.yv[i, j] + β * D.yv[i, j]
        end
    end
    return nothing
end


function update_V!(V, D, α)
    for j = 1:size(V.xc, 2)
        for i = 2:size(V.xc, 1)-1
            V.xc[i, j] += α * D.xc[i, j]
        end
    end
    for j = 2:size(V.yc, 2)-1
        for i = 1:size(V.yc, 1)
            V.yc[i, j] += α * D.yc[i, j]
        end
    end

    for j = 1:size(V.xv, 2)
        for i = 2:size(V.xv, 1)-1
            V.xv[i, j] += α * D.xv[i, j]
        end
    end

    for j = 2:size(V.yv, 2)-1
        for i = 1:size(V.yv, 1)
            V.yv[i, j] += α * D.yv[i, j]
        end
    end
    return nothing
end


function initialise_invM(invM, η, dx, dy, γ)
    nx, ny = size(η.c)

    ## inner points
    # x direction, cell centers
    for j = 2:ny-1
        for i = 2:nx
            mij = ( 2 / dx^2 * (η.c[i-1, j] + η.c[i, j  ])
                  + 1 / dy^2 * (η.v[i  , j] + η.v[i, j+1])
                  + 2 * γ / dx^2)
            invM.xc[i, j] = inv(mij)
        end
    end

    # y direction, cell centers
    for j = 2:ny
        for i = 2:nx-1
            mij = ( 2 / dy^2 * (η.c[i, j-1] + η.c[i  , j])
                  + 1 / dx^2 * (η.v[i, j  ] + η.v[i+1, j])
                  + 2 * γ / dy^2)
            invM.yc[i, j] = inv(mij)
        end
    end

    # x direction, vertices
    for j = 2:ny
        for i = 2:nx+1
            mij = ( 2 / dx^2 * (η.v[i-1, j  ] + η.v[i  , j])
                  + 1 / dy^2 * (η.c[i-1, j-1] + η.c[i-1, j])
                  + 2 * γ / dx^2)
            invM.xv[i, j] = inv(mij)
        end
    end

    # y direction, vertices
    for j=2:ny+1
        for i=2:nx
            mij = ( 2 / dy^2 * (η.v[i  , j-1] + η.v[i, j  ])
                  + 1 / dx^2 * (η.c[i-1, j-1] + η.c[i, j-1])
                  + 2 * γ / dy^2)
            invM.yv[i, j] = inv(mij)
        end
    end

    ## Neumann boundary points
    # x direction, cell centers
    for i = 2:nx
        invM.xc[i, 1 ] = inv( 2 / dx^2 * (η.c[i-1, 1] + η.c[i, 1])
                            + 1 / dy^2 * (η.v[i, 2])
                            + 2 * γ / dx^2)
        invM.xc[i, ny] = inv( 2 / dx^2 * (η.c[i-1, ny] + η.c[i, ny])
                            + 1 / dy^2 * η.v[i, ny]
                            + 2 * γ / dx^2)
    end
    # y direction, cell centers
    for j = 2:ny
        invM.yc[1 , j] = inv( 2 / dy^2 * (η.c[1, j-1] + η.c[1, j])
                            + 1 / dx^2 * (η.v[2, j])
                            + 2 * γ / dy^2)
        invM.yc[nx, j] = inv( 2 / dy^2  * (η.c[nx, j-1] + η.c[nx, j])
                            + 1 / dx^2 * (η.v[nx, j])
                            + 2 * γ / dy^2)
    end
    # x direction, vertices
    for i = 2:nx+1
        invM.xv[i, 1   ] = inv( 2 / dx^2 * (η.v[i-1, 1] + η.v[i, 1])
                              + 1 / dy^2 * (η.c[i-1, 1])
                              + 2 * γ / dx^2)
        invM.xv[i, ny+1] = inv( 2 / dx^2 * (η.v[i-1, ny+1] + η.v[i, ny+1])
                              + 1 / dy^2 * (η.c[i-1, ny  ])
                              + 2 * γ / dx^2)
    end
    # y direction, vertices
    for j = 2:ny+1
        invM.yv[1   , j] = inv( 2 / dy^2 * (η.v[1, j-1] + η.v[1, j])
                              + 1 / dx^2 * (η.c[1, j-1])
                              + 2 * γ / dy^2)
        invM.yv[nx+1, j] = inv( 2 / dy^2 * (η.v[nx+1, j-1] + η.v[nx+1, j])
                              + 1 / dx^2 * (η.c[nx  , j-1])
                              + 2 * γ / dy^2)
    end

    ## Dirichlet boundary points, leave zero

    return nothing
    
end


function nonlinear_inclusion(;n=127, ninc=5, η_ratio=0.1, niter=10000, γ_factor=1.,
                            ϵ_cg=1e-3, ϵ_ph=1e-6, ϵ_newton=1e-3, verbose=false)
    L_ref =  1. # reference length 
    ρg_avg = 1. # average density

    B_avg = 1. 
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
    ρg   = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    B    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))

    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    divV = deepcopy(P)
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

    ϵ̇_E  = zeros(nx, ny) # strain rate invariant, computed only for cell centers
    
    initialise_η_ρ!(B, ρg, B_avg, ρg_avg, η_ratio, xc, yc, xv, yv, Lx, Ly, ninc=ninc)

    # Coefficient of augmented Lagrangian
    γ = γ_factor * tplNorm(B, Inf)

    # damping parameter for newton iteration
    λ = 1.

    # residual norms for monitoring convergence
    δ = Inf # CG
    χ = Inf # Newton
    ω = Inf # Pressure

    δ_ref = tplNorm(ρg, Inf) # is this correct ?
    ω_ref = ρg_avg * Lx / B_avg

    # visualisation
    itercounts = []
    res_newton = []

    fig = Figure(size=(800,900))
    axs = (Eta=Axis(fig[1,1][1,1], aspect=1, title="viscosity (log)"),
           P =Axis(fig[1,2][1,1], aspect=1, title="pressure"),
           Vx=Axis(fig[2,1][1,1], aspect=1, title="horizontal velocity"),
           Vy=Axis(fig[2,2][1,1], aspect=1, title="vertical velocity"),
           Sr=Axis(fig[3,1][1,1], aspect=1, title="strain rate"),
           Er=Axis(fig[3,2][1,1], title="Convergence of Newton Method", xlabel="iterations / nx", ylabel="residual norm"))
    plt = (Eta=heatmap!(axs.Eta, η.c, colormap=ColorSchemes.viridis),
           P=heatmap!(axs.P, P.c, colormap=ColorSchemes.viridis),
           Vx=heatmap!(axs.Vx, V.xc, colormap=ColorSchemes.viridis),
           Vy=heatmap!(axs.Vy, V.yc, colormap=ColorSchemes.viridis),
           Sr=heatmap!(axs.Sr, ϵ̇_E, colormap=ColorSchemes.viridis))
    cbar= (Eta=Colorbar(fig[1, 1][1, 2], plt.Eta),
           P=Colorbar(fig[1, 2][1, 2], plt.P),
           Vx=Colorbar(fig[2, 1][1, 2], plt.Vx),
           Vy=Colorbar(fig[2, 2][1, 2], plt.Vy),
           Sr=Colorbar(fig[3, 1][1, 2], plt.Sr))

    display(fig)


    # Powell Hestenes
    it = 0
    while it < niter && ω > ϵ_ph
        verbose && println("Iteration ", it_P)
        tplSet!(P₀, P)

        compute_R!(R, P, η, P₀, V, ρg, B, q, ϵ̇_bg, dx, dy, γ)

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

                α = - μ / tplDot(D, Q)

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
            end
            # damped to newton iteration
            tplSet!(V̄, V)
            # tplScale!(dV, λ)
            tplAdd!(V̄, dV)
            compute_R!(R, P,  η, P₀, V̄, ρg, B, q, ϵ̇_bg, dx, dy, γ)
            χ_new = tplNorm(R, Inf) / δ_ref
            λ = 1.
            while χ_new >= χ && λ > 1e-4
                tplSet!(V̄, V)
                tplScale!(dV, inv(MathConstants.golden))
                tplAdd!(V̄, dV)
                compute_R!(R, P, η, P₀, V̄, ρg, B, q, ϵ̇_bg, dx, dy, γ)
                λ /= MathConstants.golden
                χ_new = tplNorm(R, Inf) / δ_ref
            end
            tplSet!(V, V̄)
            # λ *= MathConstants.golden
            χ = χ_new

            push!(itercounts, it)
            push!(res_newton, χ)

            # update plot
            compute_strain_rate!(ϵ̇_E, V, dx, dy, ϵ̇_bg)
            plt.Eta[3][] .= log10.(η.c)
            plt.P[3][]   .= P.c
            plt.Vx[3][]  .= V.xc
            plt.Vy[3][]  .= V.yc
            plt.Sr[3][]  .= log10.(ϵ̇_E)
            plt.Eta.colorrange[]= (min(-1,log10(minimum(η.c))), max(1,log10(maximum(η.c))))
            plt.P.colorrange[]  = (min(-1e-10,minimum(P.c)), max(1e-10, maximum(P.c)))
            plt.Vx.colorrange[] = (min(-1e-10,minimum(V.xc)), max(1e-10,maximum(V.xc)))
            plt.Vy.colorrange[] = (min(-1e-10,minimum(V.yc)), max(1e-10,maximum(V.yc)))
            plt.Sr.colorrange[]= (min(-1,log10(minimum(ϵ̇_E))), max(1,log10(maximum(ϵ̇_E))))

            scatterlines!(axs.Er, itercounts ./ nx, log10.(res_newton), color=:purple)

            display(fig)
            
            println("Newton residual = ", χ, "; λ =", λ, "; total iteration count: ", it)
        end    
        compute_divV!(divV, V, dx, dy)
        ω = tplNorm(divV, Inf) / ω_ref # correct scaling?
        println("Pressure residual = ", ω, ", Newton residual = ", χ, ", CG residual = ", δ)
    end

    return it, P, V, R, η
end


function compute_strain_rate!(ϵ̇_E, V, dx, dy, ϵ̇_bg=eps())
    nx, ny = size(ϵ̇_E)

    for j = 1:ny
        for i = 1:nx
            dVxdx = (V.xc[i+1, j] - V.xc[i, j]) / dx 
            dVydy = (V.yc[i, j+1] - V.yc[i, j]) / dy
            dVxdy_dVydx = 0.5*((V.xv[i+1, j+1] - V.xv[i+1, j]) / dy + (V.yv[i+1, j+1] - V.yv[i, j+1]) / dx)
            
            ϵ̇_E[i, j] = (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2)


        end
    end
end


n = 128
outfields = nonlinear_inclusion(n=n, ninc=4, η_ratio=10.,γ_factor=100., niter=2000*n, ϵ_ph=1e-5, ϵ_cg=1e-5, ϵ_newton=1e-5);
