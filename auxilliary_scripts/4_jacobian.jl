using Enzyme
using Random

include("../scripts/4_nonlinear_augmLagrange/square_domain_free_slip.jl")

function construct_jacobian_with_boundary(n=5, seed=1234)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    q    = 1.33
    B    = (c=ones(nx, ny), v=ones(nx+1, ny+1))
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    ρg   = deepcopy(P)
    η    = deepcopy(P)
    η̄    = deepcopy(P)
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=rand(rng, nx+1, ny  ), yc=rand(rng, nx  , ny+1),
            xv=rand(rng, nx+2, ny+1), yv=rand(rng, nx+1, ny+2))
    D    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))  # basis vector for directional derivative
    R    = deepcopy(D)  # Residuals
    Q    = deepcopy(D)  # row of Jacobian of compute_R wrt. V

    N_xc = (nx+1) * ny
    N_xv = (nx+2) * (ny+1)
    N_yc = nx * (ny+1)
    N_yv = (nx+1) * (ny+2)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = zeros(N, N) 

    E_xc = N_xc
    E_yc = E_xc + N_yc
    E_xv = E_yc + N_xv 

    col = 1
    for d = D
        for j = axes(d, 2)
            for i = axes(d, 1)
                # set one entry in search vector to 1
                d[i, j] = 1.0
                # compute the jacobian column by multiplying it with a "basis vector"
                autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q), Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀),
                        Duplicated(V, D), Const(ρg), Const(B), Const(q), Const(eps()), Const(dx), Const(dy), Const(γ))
                # store result in jacobian
                J[1:E_xc, col]      .= reshape(Q.xc, N_xc)
                J[E_xc+1:E_yc, col] .= reshape(Q.yc, N_yc)
                J[E_yc+1:E_xv, col] .= reshape(Q.xv, N_xv)
                J[E_xv+1:end, col]  .= reshape(Q.yv, N_yv)
                # increase column count
                col += 1
                # reset search vector
                D.xc .= 0.
                D.yc .= 0.
                D.xv .= 0.
                D.yv .= 0.
            end
        end
    end

    return J
end

function construct_jacobian(n=5; seed=1)
    rng = Random.MersenneTwister(seed)
    nx, ny = n, n
    dx, dy = 1/nx, 1/ny
    γ      = 5.0
    q      = 1.33
    B    = (c=ones(nx, ny), v=ones(nx+1, ny+1))
    P    = (c=zeros(nx, ny), v=zeros(nx+1, ny+1))
    ρg   = deepcopy(P)
    η    = deepcopy(P)
    η̄    = deepcopy(P)
    P₀   = deepcopy(P)  # old pressure
    P̄    = deepcopy(P)  # memory needed for auto-differentiation
    V    = (xc=rand(rng, nx+1, ny  ), yc=rand(rng, nx  , ny+1),
            xv=rand(rng, nx+2, ny+1), yv=rand(rng, nx+1, ny+2))
    D    = (xc=zeros(nx+1, ny  ), yc=zeros(nx  , ny+1),
            xv=zeros(nx+2, ny+1), yv=zeros(nx+1, ny+2))  # basis vector for directional derivative
    R    = deepcopy(D)  # Residuals
    Q    = deepcopy(D)  # row of Jacobian of compute_R wrt. V
    #V = deepcopy(D)


    N_xc = (nx-1) * ny
    N_xv = (nx) * (ny+1)
    N_yc = nx * (ny-1)
    N_yv = (nx+1) * (ny)
    N    = N_xc + N_xv + N_yc + N_yv
    J   = zeros(N, N) 

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
                autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q), Duplicated(P, P̄), Duplicated(η, η̄), Const(P₀),
                        Duplicated(V, D), Const(ρg), Const(B), Const(q), Const(eps()), Const(dx), Const(dy), Const(γ))
                # store result in jacobian
                J[1:E_xc, col]      .= reshape(Q.xc[2:end-1, :], N_xc)
                J[E_xc+1:E_yc, col] .= reshape(Q.yc[:, 2:end-1], N_yc)
                J[E_yc+1:E_xv, col] .= reshape(Q.xv[2:end-1, :], N_xv)
                J[E_xv+1:end, col]  .= reshape(Q.yv[:, 2:end-1], N_yv)
                # increase column count
                col += 1
                # reset search vector
                D.xc .= 0.
                D.yc .= 0.
                D.xv .= 0.
                D.yv .= 0.
            end
        end
    end

    return J
end
