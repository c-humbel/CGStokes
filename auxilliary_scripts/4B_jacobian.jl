using KernelAbstractions
using Enzyme
using Random
using LinearAlgebra

include("../scripts/4_nonlinear_augmLagrange/kernels_free_slip.jl")


function jvp_R!(R, dR, P, dP, τ, dτ, V, D, P₀, ρg, B, q, ϵ̇_bg, iΔx, iΔy, γ, bend, wg=64)
    nx, ny = size(P.c)
    comp_P_tau = compute_P_τ!(bend, wg, (nx+1, ny+1))
    comp_R = compute_R!(bend, wg, (nx+2, ny+2))
    autodiff(Forward, comp_P_tau, DuplicatedNoNeed(P, dP), DuplicatedNoNeed(τ, dτ),
             Const(P₀), Duplicated(V, D), Const(B),
             Const(q), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))
    autodiff(Forward, comp_R, DuplicatedNoNeed(R, dR),
             Duplicated(P, dP), Duplicated(τ, dτ), Const(ρg), Const(iΔx), Const(iΔy))
    KernelAbstractions.synchronize(bend)
end

function construct_jacobian_with_boundary(n=5, backend=CPU(), type=Float64 , seed=1234)
    rng = MersenneTwister(seed)
    nx, ny = n, n
    γ    = 5.0
    q    = 1.33
    ϵ̇_bg = eps(type)

    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny), v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    ρg   = deepcopy(P)
    B    = deepcopy(P)
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx, ny)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))
    dτ   = deepcopy(τ)
    P₀   = deepcopy(P)  # old pressure
    dP   = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny  ),
            yc=KernelAbstractions.zeros(backend, type, nx  , ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    D    = deepcopy(V)  # basis vector for directional derivative
    R    = deepcopy(V)  # Residuals
    Q    = deepcopy(V)  # row of Jacobian of compute_R wrt. V

    fill!.(values(B), 1)

    for a = V
        copyto!(a, rand(rng, size(a)...))
    end

    N_xc = length(R.xc)
    N_xv = length(R.xv)
    N_yc = length(R.yc)
    N_yv = length(R.yv)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = KernelAbstractions.zeros(backend, type, N, N)

    E_xc = N_xc
    E_yc = E_xc + N_yc
    E_xv = E_yc + N_xv

    col = 1
    for d = D
        for I = eachindex(d)
            # set one entry in search vector to 1
            d[I] = 1.0
            # compute the jacobian column by multiplying it with a "basis vector"
            jvp_R!(R, Q, P, dP, τ, dτ, V, D, P₀, ρg, B, q, ϵ̇_bg, nx, ny, γ, backend)

            # store result in jacobian
            J[1:E_xc, col]      .= reshape(Q.xc, N_xc)
            J[E_xc+1:E_yc, col] .= reshape(Q.yc, N_yc)
            J[E_yc+1:E_xv, col] .= reshape(Q.xv, N_xv)
            J[E_xv+1:end , col]  .= reshape(Q.yv, N_yv)
            # increase column count
            col += 1
            # reset search vector
            fill!.(values(D), 0)
        end
    end

    return Array(J)
end


function construct_jacobian(; n=5, backend=CPU(), type=Float64, seed=1234)
    rng = MersenneTwister(seed)
    nx, ny = n, n
    γ     = 5.0
    q    = 1.33
    ϵ̇_bg = eps(type)

    P    = (c=KernelAbstractions.zeros(backend, type, nx, ny), v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
    ρg   = deepcopy(P)
    B    = deepcopy(P)
    τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx, ny),
               yy=KernelAbstractions.zeros(backend, type, nx, ny),
               xy=KernelAbstractions.zeros(backend, type, nx, ny)),
            v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
               xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))
    dτ   = deepcopy(τ)
    P₀   = deepcopy(P)  # old pressure
    dP   = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny  ),
            yc=KernelAbstractions.zeros(backend, type, nx  , ny+1),
            xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
            yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
    D    = deepcopy(V)  # basis vector for directional derivative
    R    = deepcopy(V)  # Residuals
    Q    = deepcopy(V)  # row of Jacobian of compute_R wrt. V

    fill!.(values(B), 1)

    for a = V
        copyto!(a, rand(rng, size(a)...))
    end

    N_xc = (nx-1) * ny
    N_xv = (nx) * (ny+1)
    N_yc = nx * (ny-1)
    N_yv = (nx+1) * (ny)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = KernelAbstractions.zeros(backend, type, N, N) 

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
                jvp_R!(R, Q, P, dP, τ, dτ, V, D, P₀, ρg, B, q, ϵ̇_bg, nx, ny, γ, backend)
                # store result in jacobian
                J[1:E_xc, col]      .= reshape(Q.xc[2:end-1, :], N_xc)
                J[E_xc+1:E_yc, col] .= reshape(Q.yc[:, 2:end-1], N_yc)
                J[E_yc+1:E_xv, col] .= reshape(Q.xv[2:end-1, :], N_xv)
                J[E_xv+1:end, col]  .= reshape(Q.yv[:, 2:end-1], N_yv)
                # increase column count
                col += 1
                # reset search vector
                fill!.(values(D), 0)
            end
        end
    end

    return Array(J)
end

function construct_preconditioner_matrix(n=5, backend=CPU(), type=Float64, seed=1234)
    rng  = MersenneTwister(seed)
    γ    = 5.0
    q    = 1.33
    ϵ̇_bg = eps(type)
    B    = (c=KernelAbstractions.zeros(backend, type, n, n),
            v=KernelAbstractions.zeros(backend, type, n+1, n+1))
    ϵ̇_E  = deepcopy(B)
    V    = (xc=KernelAbstractions.zeros(backend, type, n+1, n  ),
            yc=KernelAbstractions.zeros(backend, type, n  , n+1),
            xv=KernelAbstractions.zeros(backend, type, n+2, n+1),
            yv=KernelAbstractions.zeros(backend, type, n+1, n+2))
    invM = deepcopy(V)
    
    fill!.(values(B), 1)

    for a = V
        copyto!(a, rand(rng, size(a)...))
    end

    compute_strain_rate!(backend, 64, (n+1, n+1))(ϵ̇_E, V, n, n, ϵ̇_bg)
    initialise_invM(backend, 64, (n+2, n+2))(invM, ϵ̇_E, B, q, n, n, γ)

    N_c = (n+1) * n
    N_v = (n+2) * (n+1)

    inv_m = cat(reshape(invM.xc, N_c), reshape(invM.yc, N_c),
                reshape(invM.xv, N_v), reshape(invM.yv, N_v), dims=1)
    return inv_m
end

# check that Jacobian is (almost) symmetric
Jin = construct_jacobian();

@assert all(Jin .≈ Jin')

@show maximum(abs.(Jin .- Jin'));

# check that eigenvalues are real and positive
@assert eigmin(-Jin) > 0

# compare Jacobian and preconditioner
J = construct_jacobian_with_boundary();
m = construct_preconditioner_matrix();

m_ex = zeros(size(m));
diagJ = diag(-J);
m_ex[diagJ .!= 0] = inv.(diagJ[diagJ .!= 0]);

m_ex .- m
