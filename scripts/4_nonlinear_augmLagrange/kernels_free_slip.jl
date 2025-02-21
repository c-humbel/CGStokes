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

        # Dirichlet BC
        if i == 1
            V.xc[i, j] = 0.
        end

        if i + 1 == size(V.xc, 1)
            V.xc[i+1, j] = 0.
        end

        if j == 1
            V.yc[i, j] = 0.
        end

        if j + 1 == size(V.yc, 2)
            V.yc[i, j+1] = 0.
        end

        # Pressure update
        dVxdx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy

        P.c[i, j] = P₀.c[i, j] - γ * (dVxdx + dVydy)

        # Neumann BC
        if 1 < i < size(P.c, 1) && 1 < j < size(P.c, 2)
            dVxdy_dVydx = 0.5 * ((V.xv[i+1, j+1] - V.xv[i+1, j]) * iΔy + (V.yv[i+1, j+1] - V.yv[i, j+1]) * iΔx)
        else
            dVxdy_dVydx = 0.
        end
        
        # Stress update
        η = 0.5 * B.c[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)
        τ.c.xx[i, j] = 2 * η * dVxdx
        τ.c.yy[i, j] = 2 * η * dVydy
        τ.c.xy[i, j] = 2 * η * dVxdy_dVydx
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

        η = 0.5 * B.v[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2) ^ (0.5q - 1)
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
        R.xc[i, j] = -((τ.c.xx[i, j  ] - τ.c.xx[i-1, j]) * iΔx
                     + (τ.v.xy[i, j+1] - τ.v.xy[i  , j]) * iΔy
                     - (P.c[i, j] - P.c[i-1, j]) * iΔx
                     - f.xc[i, j])
    end
    ## for velocities associated with cell corners (V.xv)
    if 1 < i < size(R.xv, 1) && j <= size(R.xv, 2)
        # all values in y direction
        # inner values in x direction

        τxy_b = j > 1 ? τ.c.xy[i-1, j-1] : 0. # zero stress at the bottom boundary
        τxy_t = j < size(R.xv, 2) ? τ.c.xy[i-1, j] : 0.  # zero stress at the top boundary

        R.xv[i, j] = -((τ.v.xx[i, j] - τ.v.xx[i-1, j]) * iΔx
                     + (τxy_t - τxy_b) * iΔy
                     - (P.v[i, j] - P.v[i-1, j]) * iΔx
                     - f.xv[i, j])
    end

    ### residual in vertical (y) direction
    ### including Neumann BC at left and right boundary
    ## for velocities associated with cell centers (V.yc)
    if i <= size(R.yc, 1) && 1 < j < size(R.yc, 2)
        # inner values in y direction
        # all values in x direction        
        R.yc[i, j] = -((τ.c.yy[i  , j] - τ.c.yy[i, j-1]) * iΔy
                     + (τ.v.xy[i+1, j] - τ.v.xy[i, j  ]) * iΔx
                     - ( P.c[i, j] -  P.c[i, j-1]) * iΔy
                     - f.yc[i, j])
    end
    ## for velocities associated with cell corners (V.yv)
    if i <= size(R.yv, 1) && 1 < j < size(R.yv, 2)
        τxy_l = i > 1 ? τ.c.xy[i-1, j-1] : 0.
        τxy_r = i < size(R.yv, 1) ? τ.c.xy[i, j-1] : 0.

        R.yv[i, j] = -((τ.v.yy[i, j] - τ.v.yy[i, j-1]) * iΔy
                     + (τxy_r - τxy_l) * iΔx
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
@kernel inbounds=true function initialise_invM(invM, V, ϵ̇_E, B, q, iΔx, iΔy, γ)
    i, j = @index(Global, NTuple)

    # x direction, cell centers
    if 1 < i < size(invM.xc, 1) && j <= size(invM.xc, 2)
        dVxdx⁻ = (V.xc[i, j] - V.xc[i-1, j]) * iΔx
        dVxdx⁺ = (V.xc[i+1, j] - V.xc[i, j]) * iΔx
        dVxdy⁻ = j > 1 ? (V.xc[i, j] - V.xc[i, j-1]) * iΔy : 0.
        dVxdy⁺ = j < size(invM.xc, 2) ? (V.xc[i, j+1] - V.xc[i, j]) * iΔy : 0.
        dVydx⁻ = (V.yc[i, j] - V.yc[i-1, j]) * iΔx
        dVydx⁺ = (V.yc[i, j+1] - V.yc[i-1, j+1]) * iΔx


        mij_1  = B.c[i-1, j] * iΔx^2 * (
                    ϵ̇_E.c[i-1, j]^(0.5q - 1)
                  + ϵ̇_E.c[i-1, j]^(0.5q - 2) * (0.5q - 1) * dVxdx⁻^2)

        mij_2  = B.c[i, j] * iΔx^2 * (
                    ϵ̇_E.c[i, j]^(0.5q - 1)
                  + ϵ̇_E.c[i, j]^(0.5q - 2) * (0.5q - 1) * dVxdx⁺^2)

        mij_3 = j > 1 ? B.v[i, j] * 0.5 * iΔy^2 * (
                    ϵ̇_E.v[i, j]^(0.5q - 1)
                  + ϵ̇_E.v[i, j]^(0.5q - 2) * (0.5q - 1) * 0.25*(dVxdy⁻ + dVydx⁻)^2
                ) : 0.

        mij_4 = j < size(invM.xc, 2) ? B.v[i, j+1] * 0.5 * iΔy^2 * (
                    ϵ̇_E.v[i, j+1]^(0.5q - 1)
                  + ϵ̇_E.v[i, j+1]^(0.5q - 2) * (0.5q - 1) * 0.25 * (dVxdy⁺ + dVydx⁺)^2
                ) : 0.
                  
        invM.xc[i, j] = inv(mij_1 + mij_2 + mij_3 + mij_4 + 2γ * iΔx^2)
    end

    # y direction, cell centers
    if i <= size(invM.yc, 1) && 1 < j < size(invM.yc, 2)
        dVydy⁻ = (V.yc[i, j] - V.yc[i-1, j]) * iΔy
        dVydy⁺ = (V.yc[i+1, j] - V.yc[i, j]) * iΔy
        dVxdy⁻ = (V.xc[i, j] - V.xc[i, j-1]) * iΔy
        dVxdy⁺ = (V.xc[i, j+1] - V.xc[i, j]) * iΔy
        dVydx⁻ = i > 1 ? (V.yc[i, j] - V.yc[i-1, j]) * iΔx : 0.
        dVydx⁺ = i < size(invM.yc, 1) ? (V.yc[i, j+1] - V.yc[i-1, j+1]) * iΔx : 0.

        mij_1  = B.c[i, j-1] * iΔy^2 * (
                    ϵ̇_E.c[i, j-1]^(0.5q - 1)
                  + ϵ̇_E.c[i, j-1]^(0.5q - 2) * (0.5q - 1) * dVydy⁻^2)

        mij_2  = B.c[i, j] * iΔy^2 * (
                    ϵ̇_E.c[i, j]^(0.5q - 1)
                  + ϵ̇_E.c[i, j]^(0.5q - 2) * (0.5q - 1) * dVydy⁺^2)

        mij_3 = i > 1 ? B.v[i, j] * 0.5 * iΔx^2 * (
                    ϵ̇_E.v[i, j]^(0.5q - 1)
                  + ϵ̇_E.v[i, j]^(0.5q - 2) * (0.5q - 1) * 0.25*(dVxdy⁻ + dVydx⁻)^2
                ) : 0.

        mij_4 = i < size(invM.yc, 1) ? B.v[i+1, j] * 0.5 * iΔx^2 * (
                    ϵ̇_E.v[i+1, j]^(0.5q - 1)
                  + ϵ̇_E.v[i+1, j]^(0.5q - 2) * (0.5q - 1) * 0.25 * (dVxdy⁺ + dVydx⁺)^2
                ) : 0.
                  
        invM.yc[i, j] = inv(mij_1 + mij_2 + mij_3 + mij_4 + 2γ * iΔy^2)
    end

    # x direction, vertices
    if 1 < i < size(invM.xv, 1) && j <= size(invM.xv, 2)
        dVxdx⁻ = (V.xv[i, j] - V.xv[i-1, j]) * iΔx
        dVxdx⁺ = (V.xv[i+1, j] - V.xv[i, j]) * iΔx
        dVxdy⁻ = j > 1 ? (V.xv[i, j] - V.xv[i, j-1]) * iΔy : 0.
        dVxdy⁺ = j < size(invM.xv, 2) ? (V.xv[i, j+1] - V.xv[i, j]) * iΔy : 0.
        dVydx⁻ = (V.yv[i, j] - V.yv[i-1, j]) * iΔx
        dVydx⁺ = (V.yv[i, j+1] - V.yv[i-1, j+1]) * iΔx


        mij_1  = B.v[i-1, j] * iΔx^2 * (
                    ϵ̇_E.v[i-1, j]^(0.5q - 1)
                  + ϵ̇_E.v[i-1, j]^(0.5q - 2) * (0.5q - 1) * dVxdx⁻^2)

        mij_2  = B.v[i, j] * iΔx^2 * (
                    ϵ̇_E.v[i, j]^(0.5q - 1)
                  + ϵ̇_E.v[i, j]^(0.5q - 2) * (0.5q - 1) * dVxdx⁺^2)

        
        mij_3 = j > 1 ? B.c[i, j-1] * 0.5 * iΔy^2 * (
                    ϵ̇_E.c[i, j-1]^(0.5q - 1)
                  + ϵ̇_E.c[i, j-1]^(0.5q - 2) * (0.5q - 1) * 0.25*(dVxdy⁻ + dVydx⁻)^2
                ) : 0.

        mij_4 = j < size(invM.xv, 2) ? B.c[i, j] * 0.5 * iΔy^2 * (
                    ϵ̇_E.c[i, j]^(0.5q - 1)
                  + ϵ̇_E.c[i, j]^(0.5q - 2) * (0.5q - 1) * 0.25 * (dVxdy⁺ + dVydx⁺)^2
                ) : 0.
                  
        invM.xv[i, j] = inv(mij_1 + mij_2 + mij_3 + mij_4 + 2γ * iΔx^2)
    end

    # y direction, vertices
    if 1 < i < size(invM.yv, 1) && 1 < j < size(invM.yv, 2)
        dVydy⁻ = (V.yv[i, j] - V.yv[i-1, j]) * iΔy
        dVydy⁺ = (V.yv[i+1, j] - V.yv[i, j]) * iΔy
        dVxdy⁻ = (V.xv[i, j] - V.xv[i, j-1]) * iΔy
        dVxdy⁺ = (V.xv[i, j+1] - V.xv[i, j]) * iΔy
        dVydx⁻ = i > 1 ? (V.yv[i, j] - V.yv[i-1, j]) * iΔx : 0.
        dVydx⁺ = i < size(invM.yv, 1) ? (V.yv[i, j+1] - V.yv[i-1, j+1]) * iΔx : 0.

        mij_1  = B.v[i, j-1] * iΔy^2 * (
                    ϵ̇_E.v[i, j-1]^(0.5q - 1)
                  + ϵ̇_E.v[i, j-1]^(0.5q - 2) * (0.5q - 1) * dVydy⁻^2)

        mij_2  = B.v[i, j] * iΔy^2 * (
                    ϵ̇_E.v[i, j]^(0.5q - 1)
                  + ϵ̇_E.v[i, j]^(0.5q - 2) * (0.5q - 1) * dVydy⁺^2)

        mij_3 = i > 1 ? B.c[i, j] * 0.5 * iΔx^2 * (
                    ϵ̇_E.c[i-1, j]^(0.5q - 1)
                  + ϵ̇_E.c[i-1, j]^(0.5q - 2) * (0.5q - 1) * 0.25*(dVxdy⁻ + dVydx⁻)^2
                ) : 0.

        mij_4 = i < size(invM.yv, 1) ? B.c[i+1, j] * 0.5 * iΔx^2 * (
                    ϵ̇_E.c[i, j]^(0.5q - 1)
                  + ϵ̇_E.c[i, j]^(0.5q - 2) * (0.5q - 1) * 0.25 * (dVxdy⁺ + dVydx⁺)^2
                ) : 0.
                  
        invM.yv[i, j] = inv(mij_1 + mij_2 + mij_3 + mij_4 + 2γ * iΔy^2)
    end

end


# dimensions for kernel launch: nx+1, ny+1
@kernel inbounds=true function compute_strain_rate!(ϵ̇_E, V, iΔx, iΔy, ϵ̇_bg)
    i, j = @index(Global, NTuple)

    if i <= size(ϵ̇_E.c, 1) && j <= size(ϵ̇_E.c, 2)
        dVxdx = (V.xc[i+1, j] - V.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] - V.yc[i, j]) * iΔy


        if 1 < i < size(ϵ̇_E.c, 1) && 1 < j < size(ϵ̇_E.c, 2)
            dVxdy_dVydx = 0.5 * ((V.xv[i+1, j+1] - V.xv[i+1, j]) * iΔy + (V.yv[i+1, j+1] - V.yv[i, j+1]) * iΔx)
        else
            dVxdy_dVydx = 0.
        end
        
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


#dimensions for kernel launch: maximum(size.(values(a), 1)), maximum(size.(values(a), 2))
@kernel inbounds=true function set_sum!(a::NamedTuple, b::NamedTuple, c::NamedTuple, α::Real=1)
    i, j = @index(Global, NTuple)

    for k = keys(a)
        if i <= size(a[k], 1) && j <=size(a[k], 2) 
            a[k][i, j] = b[k][i, j] + α * c[k][i, j]
        end
    end
end