# dimensions for kernel launch: nx+1, ny+1
@kernel inbounds=true function compute_P_τ_weighted!(P, τ, P₀, V, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
    i, j = @index(Global, NTuple)
    if i <= size(P.c, 1) && j <= size(P.c, 2)

        # Pressure update
        dVxdx = (V.xc[i+1, j] * ωₛ.xc[i+1, j] - V.xc[i, j] * ωₛ.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] * ωₛ.yc[i, j+1] - V.yc[i, j] * ωₛ.yc[i, j]) * iΔy

        P.c[i, j] = (P₀.c[i, j] - γ * (dVxdx + dVydy))
        
        # Stress update
        dVxdy_dVydx = 0.5 * ( (V.xv[i+1, j+1] * ωₛ.xv[i+1, j+1] - V.xv[i+1, j] * ωₛ.xv[i+1, j]) * iΔy
                            + (V.yv[i+1, j+1] * ωₛ.yv[i+1, j+1] - V.yv[i, j+1] * ωₛ.yv[i, j+1]) * iΔx)

        η = 0.5 * B.c[i, j] * (0.5 * dVxdx^2 + 0.5 * dVydy^2 + dVxdy_dVydx^2 + 2 * ϵ̇_bg^2)^(0.5q - 1)
        τ.c.xx[i, j] = 2 * η * dVxdx
        τ.c.yy[i, j] = 2 * η * dVydy
        τ.c.xy[i+1, j+1] =  2 * η * dVxdy_dVydx
    end

    if i <= size(P.v, 1) && j <= size(P.v, 2)        
        dVxdx = (V.xv[i+1, j] * ωₛ.xv[i+1, j] - V.xv[i, j] * ωₛ.xv[i, j]) * iΔx
        dVydy = (V.yv[i, j+1] * ωₛ.yv[i, j+1] - V.yv[i, j] * ωₛ.yv[i, j]) * iΔy

        P.v[i, j] = (P₀.v[i, j] - γ * (dVxdx + dVydy))

        if 1 < i < size(P.v, 1) && 1 < j < size(P.v, 2)
            dVxdy_dVydx = 0.5 * ( (V.xc[i, j] * ωₛ.xc[i, j] - V.xc[i, j-1] * ωₛ.xc[i, j-1]) * iΔy 
                                + (V.yc[i, j] * ωₛ.yc[i, j] - V.yc[i-1, j] * ωₛ.yc[i-1, j]) * iΔx)
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
@kernel inbounds=true function compute_R_weighted!(R, P, τ, f, ωₐ, ωₛ, iΔx, iΔy)
    i, j = @index(Global, NTuple)

    ### residual in horizontal (x) direction
    ## including Neumann BC on at top and bottom boundary
    ## for velocities associated with cell centers (V.xc)
    if 1 < i < size(R.xc, 1) && j <= size(R.xc, 2)
        # all values in y direction
        # inner values in x direction

        # residual in x direction on the interface
        R.xc[i, j] = -((τ.c.xx[i, j  ] * ωₐ.c[i+1, j+1] - τ.c.xx[i-1, j] * ωₐ.c[i, j+1]) * iΔx
                     + (τ.v.xy[i, j+1] * ωₐ.v[i  , j+1] - τ.v.xy[i  , j] * ωₐ.v[i, j  ]) * iΔy
                     - (P.c[i, j] * ωₐ.c[i+1, j+1] - P.c[i-1, j]* ωₐ.c[i, j+1]) * iΔx
                     - f.xc[i, j] * ωₐ.xc[i, j]) * ωₛ.xc[i, j]
    end
    ## for velocities associated with cell corners (V.xv)
    if 1 < i < size(R.xv, 1) && j <= size(R.xv, 2)
        # all values in y direction
        # inner values in x direction
        R.xv[i, j] = -((τ.v.xx[i, j  ] * ωₐ.v[i, j  ] - τ.v.xx[i-1, j] * ωₐ.v[i-1, j]) * iΔx
                     + (τ.c.xy[i, j+1] * ωₐ.c[i, j+1] - τ.c.xy[i  , j] * ωₐ.c[i  , j]) * iΔy
                     - (P.v[i, j] * ωₐ.v[i, j] - P.v[i-1, j] * ωₐ.v[i-1, j]) * iΔx
                     - f.xv[i, j] * ωₐ.xv[i, j]) * ωₛ.xv[i, j]
    end

    ### residual in vertical (y) direction
    ### including Neumann BC at left and right boundary
    ## for velocities associated with cell centers (V.yc)
    if i <= size(R.yc, 1) && 1 < j < size(R.yc, 2)
        # inner values in y direction
        # all values in x direction        
        R.yc[i, j] = -((τ.c.yy[i  , j] * ωₐ.c[i+1, j+1] - τ.c.yy[i, j-1] * ωₐ.c[i+1, j]) * iΔy
                     + (τ.v.xy[i+1, j] * ωₐ.v[i+1, j  ] - τ.v.xy[i, j  ] * ωₐ.v[i  , j]) * iΔx
                     - (P.c[i, j] * ωₐ.c[i+1, j+1] -  P.c[i, j-1] * ωₐ.c[i+1, j]) * iΔy
                     - f.yc[i, j] * ωₐ.yc[i, j]) * ωₛ.yc[i, j]
    end
    ## for velocities associated with cell corners (V.yv)
    if i <= size(R.yv, 1) && 1 < j < size(R.yv, 2)
        R.yv[i, j] = -((τ.v.yy[i  , j] * ωₐ.v[i  , j] - τ.v.yy[i, j-1] * ωₐ.v[i, j-1]) * iΔy
                     + (τ.c.xy[i+1, j] * ωₐ.c[i+1, j] - τ.c.xy[i, j  ] * ωₐ.c[i, j  ]) * iΔx
                     - (P.v[i, j] * ωₐ.v[i, j] -  P.v[i, j-1] * ωₐ.v[i, j-1]) * iΔy
                     - f.yv[i, j] * ωₐ.yv[i, j]) * ωₛ.yv[i, j]
    end

    # Residuals corresponding to cells affected by Dirichlet BC are left zero
end


@kernel inbounds=true function compute_divV_weighted!(divV, V, ωₐ, ωₛ, iΔx, iΔy)
    i, j = @index(Global, NTuple)
    
    if i < size(V.xc, 1) && j < size(V.yc, 2)
        dVxdx = (V.xc[i+1, j] * ωₛ.xc[i+1, j] - V.xc[i, j] * ωₛ.xc[i, j]) * iΔx 
        dVydy = (V.yc[i, j+1] * ωₛ.yc[i, j+1] - V.yc[i, j] * ωₛ.yc[i, j]) * iΔy
        divV.c[i, j] = ωₐ.xc[i+1, j] * ωₐ.xc[i, j] * ωₐ.yc[i, j+1] * ωₐ.yc[i, j] * (dVxdx + dVydy)
    end

    if i < size(V.xv, 1) && j < size(V.yv, 2)
        dVxdx = (V.xv[i+1, j] * ωₛ.xv[i+1, j] - V.xv[i, j] * ωₛ.xv[i, j]) * iΔx
        dVydy = (V.yv[i, j+1] * ωₛ.yv[i, j+1] - V.yv[i, j] * ωₛ.yv[i, j]) * iΔy
        divV.v[i, j] = ωₐ.xv[i+1, j] * ωₐ.xv[i, j] * ωₐ.yv[i, j+1] * ωₐ.yv[i, j] * (dVxdx + dVydy)
    end
end


@views function initialise_volume_fractions_ring_segment!(ωₐ, ωₛ,  x₀, y₀, rₐ, rₛ, xc, yc, xv, yv)
    copyto!(ωₐ.c[2:end-1, 2:end-1], [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xc, y=yc])
    ωₐ.c[[1, end], :] .= ωₐ.c[[2, end-1], :]
    ωₐ.c[:, [1, end]] .= ωₐ.c[:, [2, end-1]]
    copyto!(ωₐ.v , [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xv, y=yv])
    copyto!(ωₐ.xc, [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xv, y=yc])
    copyto!(ωₐ.yc, [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xc, y=yv])
    copyto!(ωₐ.xv[2:end-1, :], [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xc, y=yv])
    copyto!(ωₐ.yv[:, 2:end-1], [(x - x₀)^2 + (y - y₀)^2 < rₐ^2 ? 1. : 0. for x=xv, y=yc])
    ωₐ.xv[[1, end], :] .= ωₐ.xv[[2, end-1], :]
    ωₐ.yv[:, [1, end]] .= ωₐ.yv[:, [2, end-1]]

    copyto!(ωₛ.c[2:end-1, 2:end-1], [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xc, y=yc])
    ωₛ.c[[1, end], :] .= ωₛ.c[[2, end-1], :]
    ωₛ.c[:, [1, end]] .= ωₛ.c[:, [2, end-1]]
    copyto!(ωₛ.v,  [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xv, y=yv])
    copyto!(ωₛ.xc, [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xv, y=yc])
    copyto!(ωₛ.yc, [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xc, y=yv])
    copyto!(ωₛ.xv[2:end-1, :], [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xc, y=yv])
    copyto!(ωₛ.yv[:, 2:end-1], [(x - x₀)^2 + (y - y₀)^2 < rₛ^2 ? 0. : 1. for x=xv, y=yc])
    ωₛ.xv[[1, end], :] .= ωₛ.xv[[2, end-1], :]
    ωₛ.yv[:, [1, end]] .= ωₛ.yv[:, [2, end-1]]
    return nothing
end

@views function initialise_volume_fractions_from_function!(ω, fun, xc, yc, xv, yv, v_below, v_above)
    copyto!(ω.c[2:end-1, 2:end-1], [y <= fun(x) ? v_below : v_above for x=xc, y=yc])
    ω.c[[1, end], :] .= ω.c[[2, end-1], :]
    ω.c[:, [1, end]] .= ω.c[:, [2, end-1]]
    copyto!(ω.v , [y <= fun(x) ? v_below : v_above  for x=xv, y=yv])
    copyto!(ω.xc, [y <= fun(x) ? v_below : v_above  for x=xv, y=yc])
    copyto!(ω.yc, [y <= fun(x) ? v_below : v_above  for x=xc, y=yv])
    copyto!(ω.xv[2:end-1, :], [y <= fun(x) ? v_below : v_above  for x=xc, y=yv])
    copyto!(ω.yv[:, 2:end-1], [y <= fun(x) ? v_below : v_above  for x=xv, y=yc])
    ω.xv[[1, end], :] .= ω.xv[[2, end-1], :]
    ω.yv[:, [1, end]] .= ω.yv[:, [2, end-1]]
end

@views function initialise_f_inclusions!(f, xi, yi, ri, ρgi, ρg_b, xc, yc, xv, yv)
    fill!(f.xc, 0)
    fill!(f.xv, 0)  

    fyc = Array(f.yc)
    fyv = Array(f.yv)
    fill!(fyc, ρg_b)  
    fill!(fyv, ρg_b)  
    for (_xi, _yi, _ri, _ρgi) = zip(xi, yi, ri, ρgi)
        for j = eachindex(yv)
            for i = eachindex(xc)
                if (xc[i] - _xi)^2 + (yv[j] - _yi)^2 < _ri^2
                    fyc[i, j] = _ρgi

                end
            end
        end
        for j = eachindex(yc)
            for i = eachindex(xv)
                if (xv[i] - _xi)^2 + (yc[j] - _yi)^2 < _ri^2
                    fyv[i, j] = _ρgi
                end
            end
        end
    end
    copyto!(f.yc, fyc)
    copyto!(f.yv, fyv)
    return nothing
end