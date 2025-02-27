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
end


# dimensions for kernel launch: nx+1, ny+1
@kernel inbounds=true function compute_P_τ!(P, τ, P₀, V, B, q, ϵ̇_bg, iΔx, iΔy, γ)
    i, j = @index(Global, NTuple)
    if i <= size(P.c, 1) && j <= size(P.c, 2)

        # BC
        if i == 1
            V.xc[i, j] = 0.
            τ.c.xx[i, j+1] = 0.
            τ.c.yy[i, j+1] = 0.
            τ.c.xy[i, j+1] = 0.
        end

        if i + 1 == size(V.xc, 1)
            V.xc[i+1, j] = 0.
            τ.c.xx[i+2, j+1] = 0.
            τ.c.yy[i+2, j+1] = 0.
            τ.c.xy[i+2, j+1] = 0.
        end

        if j == 1
            V.yc[i, j] = 0.
            τ.c.xx[i+1, j] = 0.
            τ.c.yy[i+1, j] = 0.
            τ.c.xy[i+1, j] = 0.
        end

        if j + 1 == size(V.yc, 2)
            V.yc[i, j+1] = 0.
            τ.c.xx[i+1, j+2] = 0.
            τ.c.yy[i+1, j+2] = 0.
            τ.c.xy[i+1, j+2] = 0.
        end

        # Pressure update
        dVxdx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy

        P.c[i, j] = P₀.c[i, j] - γ * (dVxdx + dVydy)
        
        # Stress update
        dVxdy_dVydx = 0.5 * ((V.xv[i+1, j+1] - V.xv[i+1, j]) * iΔy + (V.yv[i+1, j+1] - V.yv[i, j+1]) * iΔx)

        η = 0.5 * B.c[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2)^(0.5q - 1)
        τ.c.xx[i+1, j+1] = 2 * η * dVxdx
        τ.c.yy[i+1, j+1] = 2 * η * dVydy
        τ.c.xy[i+1, j+1] = 2 * η * dVxdy_dVydx
    end

    if i <= size(P.v, 1) && j <= size(P.v, 2)
        if i == 1
            V.xv[i, j] = 0.
        end

        if i + 1 == size(V.xv, 1)
            V.xv[i+1, j] = 0.
        end

        if j == 1
            V.yv[i, j] = 0.
        end

        if j + 1 == size(V.yv, 2)
            V.yv[i, j+1] = 0.
        end
        
        dVxdx = (V.xv[i+1, j] - V.xv[i, j]) * iΔx
        dVydy = (V.yv[i, j+1] - V.yv[i, j]) * iΔy

        P.v[i, j] = P₀.v[i, j] - γ * (dVxdx + dVydy)

        if 1 < i < size(P.v, 1) && 1 < j < size(P.v, 2)
            dVxdy_dVydx = 0.5 * ((V.xc[i, j] - V.xc[i, j-1]) * iΔy + (V.yc[i, j] - V.yc[i-1, j]) * iΔx)
        else
            dVxdy_dVydx = 0.
        end

        η = 0.5 * B.v[i, j] * ((0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1))
        τ.v.xx[i, j] = 2 * η * dVxdx
        τ.v.yy[i, j] = 2 * η * dVydy
        τ.v.xy[i, j] = 2 * η * dVxdy_dVydx
    end
end


# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function compute_R!(R, P, τ, f, iΔx, iΔy)
    i, j = @index(Global, NTuple)

    ### residual in horizontal (x) direction
    ## including Neumann BC on at top and bottom boundary
    ## for velocities associated with cell centers (V.xc)
    if 1 < i < size(R.xc, 1) && j <= size(R.xc, 2)
        # all values in y direction
        # inner values in x direction

        # residual in x direction on the interface
        R.xc[i, j] = -( (τ.c.xx[i+1, j+1] - τ.c.xx[i, j+1]) * iΔx
                     + (τ.v.xy[i, j+1] - τ.v.xy[i  , j]) * iΔy
                     - (P.c[i, j] - P.c[i-1, j]) * iΔx
                     - f.xc[i, j])
    end
    ## for velocities associated with cell corners (V.xv)
    if 1 < i < size(R.xv, 1) && j <= size(R.xv, 2)
        # all values in y direction
        # inner values in x direction
        R.xv[i, j] = -( (τ.v.xx[i, j] - τ.v.xx[i-1, j]) * iΔx
                     + (τ.c.xy[i, j+1] - τ.c.xy[i, j]) * iΔy
                     - (P.v[i, j] - P.v[i-1, j]) * iΔx
                     - f.xv[i, j])
    end

    ### residual in vertical (y) direction
    ### including Neumann BC at left and right boundary
    ## for velocities associated with cell centers (V.yc)
    if i <= size(R.yc, 1) && 1 < j < size(R.yc, 2)
        # inner values in y direction
        # all values in x direction        
        R.yc[i, j] = -( (τ.c.yy[i+1, j+1] - τ.c.yy[i+1, j]) * iΔy
                     + (τ.v.xy[i+1, j] - τ.v.xy[i, j  ]) * iΔx
                     - ( P.c[i, j] -  P.c[i, j-1]) * iΔy
                     - f.yc[i, j])
    end
    ## for velocities associated with cell corners (V.yv)
    if i <= size(R.yv, 1) && 1 < j < size(R.yv, 2)
        R.yv[i, j] = -( (τ.v.yy[i, j] - τ.v.yy[i, j-1]) * iΔy
                     + ( τ.c.xy[i+1, j]  - τ.c.xy[i, j]) * iΔx
                     - ( P.v[i, j] -  P.v[i, j-1]) * iΔy
                     - f.yv[i, j])
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
end


# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function update_D!(D, R, invM, β)
    i, j = @index(Global, NTuple)
    if 1 < i < size(D.xc, 1) && j <= size(D.xc, 2)
        D.xc[i, j] = invM.xc[i, j] * R.xc[i, j] + β * D.xc[i, j]
    end

    if i <= size(D.yc, 1) && 1 < j < size(D.yc, 2)
        D.yc[i, j] = invM.yc[i, j] * R.yc[i, j] + β * D.yc[i, j]
    end

    if 1 < i < size(D.xv, 1) && j <= size(D.xv, 2)
        D.xv[i, j] = invM.xv[i, j] * R.xv[i, j] + β * D.xv[i, j]
    end

    if i <= size(D.yv, 1) && 1 < j < size(D.yv, 2)
        D.yv[i, j] = invM.yv[i, j] * R.yv[i, j] + β * D.yv[i, j]
    end
end


# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function update_V!(V, D, α)
    i, j = @index(Global, NTuple)
    if 1 < i < size(V.xc, 1) && j <= size(V.xc, 2)
        V.xc[i, j] += α * D.xc[i, j]
    end

    if i <= size(V.yc, 1) && 1 < j < size(V.yc, 2)
        V.yc[i, j] += α * D.yc[i, j]
    end

    if 1 < i < size(V.xv, 1) && j <= size(V.xv, 2)
        V.xv[i, j] += α * D.xv[i, j]
    end

    if i <= size(V.yv, 1) && 1 < j < size(V.yv, 2)
        V.yv[i, j] += α * D.yv[i, j]
    end
end


# dimensions for kernel launch: nx+2, ny+2
@kernel inbounds=true function initialise_invM(invM, ϵ̇_E, B, q, iΔx, iΔy, γ)
    i, j = @index(Global, NTuple)

    ηc(i, j) = 0.5*B.c[i, j] * (ϵ̇_E.c[i, j] ^ (0.5q - 1))
    ηv(i, j) = 0.5*B.v[i, j] * (ϵ̇_E.v[i, j] ^ (0.5q - 1))
    ## inner points
    # x direction, cell centers
    if 1 < i < size(invM.xc, 1) && 1 < j < size(invM.xc, 2)
        mij = ( 2iΔx^2 * (ηc(i-1, j) + ηc(i, j))
               + iΔy^2 * (ηv(i, j) + ηv(i, j+1))
               + 2γ * iΔx^2)
        invM.xc[i, j] = inv(mij)
    end

    # y direction, cell centers
    if 1 < i < size(invM.yc, 1) && 1 < j < size(invM.yc, 2)
        mij = ( 2iΔy^2 * (ηc(i, j-1) + ηc(i, j))
               + iΔx^2 * (ηv(i, j) + ηv(i+1, j))
               + 2γ * iΔy^2)
        invM.yc[i, j] = inv(mij)
    end

    # x direction, vertices
    if 1 < i < size(invM.xv, 1) && 1 < j < size(invM.xv, 2)
        mij = ( 2iΔx^2 * (ηv(i-1, j) + ηv(i, j))
               + iΔy^2 * (ηc(i-1, j-1) + ηc(i-1, j))
               + 2γ * iΔx^2)
        invM.xv[i, j] = inv(mij)
    end

    # y direction, vertices
    if 1 < i < size(invM.yv, 1) && 1 < j < size(invM.yv, 2)
        mij = ( 2iΔy^2 * (ηv(i, j-1) + ηv(i, j))
               + iΔx^2 * (ηc(i-1, j-1) + ηc(i, j-1))
               + 2γ * iΔy^2)
        invM.yv[i, j] = inv(mij)
    end

    ## Neumann boundary points
    # x direction, cell centers
    if 1 < i < size(invM.xc, 1)
        if j == 1
            invM.xc[i, j] = inv(2iΔx^2 * (ηc(i-1, j) + ηc(i, j))
                               + iΔy^2 * (ηv(i, j+1))
                               + 2γ * iΔx^2)
        elseif j == size(invM.xc, 2)
            invM.xc[i, j] = inv(2iΔx^2 * (ηc(i-1, j) + ηc(i, j))
                               + iΔy^2 * (ηv(i  , j))
                               + 2γ * iΔx^2)
        end
    end
    # y direction, cell centers
    if 1 < j < size(invM.yc, 2)
        if i == 1
            invM.yc[i, j] = inv(2iΔy^2 * (ηc(i, j-1) + ηc(i, j))
                               + iΔx^2 * (ηv(i+1, j))
                               + 2γ * iΔy^2)
        end
        if i == size(invM.yc, 1)
            invM.yc[i, j] = inv(2iΔy^2 * (ηc(i, j-1) + ηc(i, j))
                               + iΔx^2 * (ηv(i, j  ))
                               + 2γ * iΔy^2)
        end
    end
    # x direction, vertices
    if 1 < i < size(invM.xv, 1)
        if j == 1
            invM.xv[i, j] = inv(2iΔx^2 * (ηv(i-1, j) + ηv(i, j))
                               + iΔy^2 * (ηc(i-1, j))
                               + 2γ * iΔx^2)
        end
        if j == size(invM.xv, 2)
            invM.xv[i, j] = inv(2iΔx^2 * (ηv(i-1, j) + ηv(i, j))
                               + iΔy^2 * (ηc(i-1, j-1))
                               + 2γ * iΔx^2)
        end
    end
    # y direction, vertices
    if 1 < j < size(invM.yv, 2)
        if i == 1
            invM.yv[i, j] = inv(2iΔy^2 * (ηv(i, j-1) + ηv(i, j))
                               + iΔx^2 * (ηc(i, j-1))
                               + 2γ * iΔy^2)
        end
        if i == size(invM.yv, 1)
            invM.yv[i, j] = inv(2iΔy^2 * (ηv(i  , j-1) + ηv(i, j))
                               + iΔx^2 * (ηc(i-1, j-1))
                               + 2γ * iΔy^2)
        end
    end

    ## Dirichlet boundary points, leave zero
end


# dimensions for kernel launch: nx+1, ny+1
@kernel inbounds=true function compute_strain_rate!(ϵ̇_E, V, iΔx, iΔy, ϵ̇_bg)
    i, j = @index(Global, NTuple)

    if i <= size(ϵ̇_E.c, 1) && j <= size(ϵ̇_E.c, 2)
        dVxdx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy
        dVxdy_dVydx = 0.5 * ((V.xv[i+1, j+1] - V.xv[i+1, j]) * iΔy + (V.yv[i+1, j+1] - V.yv[i, j+1]) * iΔx)
        
        ϵ̇_E.c[i, j] = (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2)
    end

    if i <= size(ϵ̇_E.v, 1) && j <= size(ϵ̇_E.v, 2)
        dVxdx = (V.xv[i+1, j] - V.xv[i, j]) * iΔx
        dVydy = (V.yv[i, j+1] - V.yv[i, j]) * iΔy

        if 1 < i < size(ϵ̇_E.v, 1) && 1 < j < size(ϵ̇_E.v, 2)
            dVxdy_dVydx = 0.5 * ((V.xc[i, j] - V.xc[i, j-1]) * iΔy + (V.yc[i, j] - V.yc[i-1, j]) * iΔx)
        else
            dVxdy_dVydx = 0.
        end

        ϵ̇_E.v[i, j] = (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2)
    end
end


@kernel inbounds=true function compute_K!(K, R, Q)
    i, j = @index(Global, NTuple)
    if i <= size(K.xc, 1) && j <= size(K.xc, 2)
        K.xc[i, j] = R.xc[i, j] - Q.xc[i, j]
    end

    if i <= size(K.yc, 1) && j <= size(K.yc, 2)
        K.yc[i, j] = R.yc[i, j] - Q.yc[i, j]
    end

    if i <= size(K.xv, 1) && j <= size(K.xv, 2)
        K.xv[i, j] = R.xv[i, j] - Q.xv[i, j]
    end

    if i <= size(K.yv, 1) && j <= size(K.yv, 2)
        K.yv[i, j] = R.yv[i, j] - Q.yv[i, j]
    end
end

@kernel inbounds=true function update_K!(K, Q, α)
    i, j = @index(Global, NTuple)
    if 1 < i < size(K.xc, 1) && j <= size(K.xc, 2)
        K.xc[i, j] -= α * Q.xc[i, j]
    end

    if i <= size(K.yc, 1) && 1 < j < size(K.yc, 2)
        K.yc[i, j] -= α * Q.yc[i, j]
    end

    if 1 < i < size(K.xv, 1) && j <= size(K.xv, 2)
        K.xv[i, j] -= α * Q.xv[i, j]
    end

    if i <= size(K.yv, 1) && 1 < j < size(K.yv, 2)
        K.yv[i, j] -= α * Q.yv[i, j]
    end
end


@kernel inbounds=true function try_step_V!(V̄, V, dV, λ)
    i, j = @index(Global, NTuple)
    if i <= size(V̄.xc, 1) && j <= size(V̄.xc, 2)
        V̄.xc[i, j] = V.xc[i, j] - λ * dV.xc[i, j]
    end

    if i <= size(V̄.yc, 1) && j <= size(V̄.yc, 2)
        V̄.yc[i, j] = V.yc[i, j] - λ * dV.yc[i, j]
    end

    if i <= size(V̄.xv, 1) && j <= size(V̄.xv, 2)
        V̄.xv[i, j] = V.xv[i, j] - λ * dV.xv[i, j]
    end

    if i <= size(V̄.yv, 1) && j <= size(V̄.yv, 2)
        V̄.yv[i, j] = V.yv[i, j] - λ * dV.yv[i, j]
    end

end


@kernel inbounds=true function initialise_D!(D, K, invM)
    i, j = @index(Global, NTuple)
    if i <= size(D.xc, 1) && j <= size(D.xc, 2)
        D.xc[i, j] = invM.xc[i, j] * K.xc[i, j]
    end

    if i <= size(D.yc, 1) && j <= size(D.yc, 2)
        D.yc[i, j] = invM.yc[i, j] * K.yc[i, j]
    end

    if i <= size(D.xv, 1) && j <= size(D.xv, 2)
        D.xv[i, j] = invM.xv[i, j] * K.xv[i, j]
    end

    if i <= size(D.yv, 1) && j <= size(D.yv, 2)
        D.yv[i, j] = invM.yv[i, j] * K.yv[i, j]
    end
end


@kernel inbounds=true function assign_flux_field!(this, other)
    i, j = @index(Global, NTuple)
    if i <= size(this.xc, 1) && j <= size(this.xc, 2)
        this.xc[i, j] = other.xc[i, j]
    end

    if i <= size(this.yc, 1) && j <= size(this.yc, 2)
        this.yc[i, j] = other.yc[i, j]
    end

    if i <= size(this.xv, 1) && j <= size(this.xv, 2)
        this.xv[i, j] = other.xv[i, j]
    end

    if i <= size(this.yv, 1) && j <= size(this.yv, 2)
        this.yv[i, j] = other.yv[i, j]
    end
    
end


@kernel inbounds=true function set_part_to_ones!(V̄, even, comp)
    i, j = @index(Global, NTuple)
    set_this = (even && (i+j) % 2 == 0) || (!even && (i+j) % 2 == 1)
    if i <= size(V̄.xc, 1) && j <= size(V̄.xc, 2)
        if comp == :xc
            V̄.xc[i, j] = set_this ? 1. : 0.
        else
            V̄.xc[i, j] = 0.
        end
    end
    if i <= size(V̄.yc, 1) && j <= size(V̄.yc, 2)
        if comp == :yc
            V̄.yc[i, j] = set_this ? 1. : 0.
        else
            V̄.yc[i, j] = 0.
        end
    end
    if i <= size(V̄.xv, 1) && j <= size(V̄.xv, 2)
        if comp == :xv
            V̄.xv[i, j] = set_this ? 1. : 0.
        else
            V̄.xv[i, j] = 0.
        end
    end
    if i <= size(V̄.yv, 1) && j <= size(V̄.yv, 2)
       if comp == :yv
            V̄.yv[i, j] = set_this ? 1. : 0.
        else
            V̄.yv[i, j] = 0.
        end
    end
end


@kernel inbounds=true function assign_part!(dest, src, even)
    i, j = @index(Global, NTuple)
    if i <= size(dest, 1) && j <= size(dest, 2)
        if (even && (i+j) % 2 == 0) || (!even && (i+j) % 2 == 1)
            dest[i, j] = src[i, j] 
        end
    end
end


@kernel inbounds=true function invert!(M)
    i, j = @index(Global, NTuple)
    if i <= size(M.xc, 1) && j <= size(M.xc, 2)
        M.xc[i, j] = M.xc[i, j] != 0. ? inv(M.xc[i, j]) : 0.
    end
    if i <= size(M.yc, 1) && j <= size(M.yc, 2)
        M.yc[i, j] = M.yc[i, j] != 0. ? inv(M.yc[i, j]) : 0.
    end
    if i <= size(M.xv, 1) && j <= size(M.xv, 2)
        M.xv[i, j] = M.xv[i, j] != 0. ? inv(M.xv[i, j]) : 0.
    end
    if i <= size(M.yv, 1) && j <= size(M.yv, 2)
        M.yv[i, j] = M.yv[i, j] != 0. ? inv(M.yv[i, j]) : 0.
    end
end