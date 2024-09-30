using CairoMakie
using LinearAlgebra

function update_P_τ!(Pt, τ, divV, V, η, dx, dy, κΔθ)
    nx, ny = size(Pt)
    for j = 1:ny
        for i = 1:nx
            dVx = (V.x[i+1, j] - V.x[i, j]) / dx
            dVy = (V.y[i, j+1] - V.y[i, j]) / dy
            divV[i, j] = dVx + dVy
            Pt[i, j]  -= κΔθ[i, j] * divV[i, j]
            τ.xx[i, j] = 2 * η[i, j] * (dVx - 1/3 * divV[i, j])
            τ.yy[i, j] = 2 * η[i, j] * (dVy - 1/3 * divV[i, j])
            if i < nx && j < ny
                # average η to cell corner
                η_c = 0.25 * (η[i, j] + η[i+1, j] + η[i, j+1] + η[i+1, j+1])
                τ.xy[i, j] = η_c * ((V.x[i+1, j+1] - V.x[i+1, j]) / dy
                                    + (V.y[i+1, j+1] - V.y[i, j+1]) / dx)
            end
        end
    end
    return nothing
end


function update_dV!(dV, R, Pt, τ, ρg, dx, dy, damp)
    nx, ny = size(Pt)
    # update velocity update in x
    for j = 1:ny-2
        for i = 1:nx-1
            R.x[i, j]  = ((τ.xx[i+1, j+1] - τ.xx[i, j+1]) / dx
                        + (τ.xy[i, j+1] - τ.xy[i, j]) / dy
                        - (Pt[i+1, j+1] - Pt[i, j+1]) / dx)
            dV.x[i, j] = damp.x * dV.x[i, j] + R.x[i, j]
        end
    end

    for j = 1:ny-1
        for i = 1:nx-2
            R.y[i, j]  = ((τ.yy[i+1, j+1] - τ.yy[i+1, j]) / dy
                        + (τ.xy[i+1, j] - τ.xy[i, j]) / dx
                        - (Pt[i+1, j+1] - Pt[i+1, j]) / dy
                        + (ρg[i+1, j+1] + ρg[i+1, j]) * 0.5)
            dV.y[i, j] = damp.y * dV.y[i, j] + R.y[i, j]
        end
    end
    return nothing
end


function update_V!(V, dV, Δθ_ρx, Δθ_ρy)
    for j = 2:size(V.x, 2)-1
        for i = 2:size(V.x, 1)-1
            V.x[i, j] += Δθ_ρx[i-1, j-1] * dV.x[i-1, j-1]
        end
    end
    for j = 2:size(V.y, 2)-1
        for i = 2:size(V.y, 1)-1
            V.y[i, j] += Δθ_ρy[i-1, j-1] * dV.y[i-1, j-1]
        end
    end
    return nothing
end


function bc_x!(f)
    for j = axes(f, 2)
        f[1,   j] = f[2,     j]
        f[end, j] = f[end-1, j]
    end
    return nothing
end

function bc_y!(f)
    for i = axes(f, 1)
        f[i, 1  ] = f[i, 2    ]
        f[i, end] = f[i, end-1]
    end
    return nothing
end


function linearStokes2D(η_ratio=0.1)
    Lx = Ly = 10.
    R_in  = 1.
    η_out = 1.
    η_in  = η_ratio * η_out
    ρg_in = 1.

    nx = ny = 127
    max_iter = 1000000
    max_err  = 1e-6
    ncheck = 5000

    dx, dy = Lx / nx, Ly / ny
    xs = LinRange(-0.5Lx + 0.5dx, 0.5Lx - 0.5dx, nx)
    ys = LinRange(-0.5Ly + 0.5dy, 0.5Ly - 0.5dy, ny)

    # field initialisation
    η    = [x^2 + y^2 < R_in^2 ? η_in : η_out for x=xs, y=ys]
    ρg   = [x^2 + y^2 < R_in^2 ? ρg_in : 0. for x=xs, y=ys]
    Pt   = zeros(nx, ny)
    divV = zeros(nx, ny)
    V    = (x=zeros(nx+1, ny), y=zeros(nx, ny+1))
    dV   = (x=zeros(nx-1, ny-2), y=zeros(nx-2, ny-1))
    R    = (x=zeros(nx-1, ny-2), y=zeros(nx-2, ny-1))
    τ    = (xx=zeros(nx, ny), xy=zeros(nx-1, ny-1), yy=zeros(nx, ny))

    # numerical parameters according to the Stokes2D miniapp in ParallelStencil.jl
    #  - change prefactor 4.1 to 5.1
    #  - replace viscosity average by maximum in domain of dependence
    #  - for pressure, use viscosity minimum in domain of dependence
    preθv = min(dx, dy)^2 / 5.1
    preθp = 0.25 * 5.1 / max(nx, ny)
    Δθ_ρx = [preθv / max(η[i, j], η[i+1, j], η[i, j+1], η[i+1, j+1], η[i, j+2], η[i+1, j+2])
             for i=1:nx-1, j=1:ny-2]
    Δθ_ρy = [preθv / max(η[i, j], η[i+1, j], η[i+2, j], η[i, j+1], η[i+1, j+1], η[i+2, j+1])
             for i=1:nx-2, j=1:ny-1]
    κΔθ   = preθp * η
    for j=2:ny-1
        for i=2:nx-1
            κΔθ[i, j] = preθp * min(η[i, j], η[i-1, j], η[i+1, j], η[i, j-1], η[i, j+1])
        end
    end
    
    damp = (x=1-4/nx, y=1-4/ny)

    # visualisation
    fig = Figure()
    axs = (Pt=Axis(fig[1,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Pressure"),
           η=Axis(fig[1,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Viscosity"),
           Vy=Axis(fig[2,1][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity"),
           Ry=Axis(fig[2,2][1,1], aspect=1, xlabel="x", ylabel="y", title="Vertical Velocity Residual"))
    plt = (Pt=image!(axs.Pt, (xs[1], xs[end]), (ys[1], ys[end]), Pt, colormap=:inferno),
           η=image!(axs.η, (xs[1], xs[end]), (ys[1], ys[end]), η, colormap=:inferno),
           Vy=image!(axs.Vy, (xs[1], xs[end]), (ys[1]-dy/2, ys[end]+dy/2), V.y, colormap=:inferno),
           Ry=image!(axs.Ry, (xs[1]+dx, xs[end]-dx), (ys[1]+dy/2, ys[end]-dy/2), R.y, colormap=:inferno))
    Colorbar(fig[1, 1][1, 2], plt.Pt)
    Colorbar(fig[1, 2][1, 2], plt.η)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Ry)

    # loop
    err = 2 * max_err
    it = 1
    while it <= max_iter && err > max_err
        update_P_τ!(Pt, τ, divV, V, η, dx, dy, κΔθ)
        update_dV!(dV, R, Pt, τ, ρg, dx, dy, damp)
        update_V!(V, dV, Δθ_ρx, Δθ_ρy)
        bc_y!(V.x)
        bc_x!(V.y)
        if it % ncheck == 0
            err = max(norm(divV, 1)/length(divV), norm(R.x, 1)/length(R.x), norm(R.y, 1)/length(R.y))
            
            println("iteration ", it, ": err = ", err)
            
        end
        it += 1
    end

    println("total number of iterations: ", it)
    plt.Pt[3] = Pt
    plt.Vy[3] = V.y
    plt.Ry[3] = log10.(abs.(R.y))
    display(fig)

    return nothing

end

linearStokes2D(1e-9)