using Enzyme, LinearAlgebra

function E(u, ∇u, dx)
    u[1] = 0.0
    u[end] = 1.0
    @. ∇u = (u[2:end] - u[1:end-1]) / dx
    return 0.5 * sum(∇u .^ 2)
end

function residual!(r, u, q, dx)
    nx = length(u)

    # r[1] = u[1]
    # r[1] = -(-(u[2] - u[1]) / dx + (u[1] + u[2]) / dx) / dx
    # r[end] = u[end] - 1.0

    u[1] = 0.0
    u[end] = 1.0

    # q[1] = -(u[1] - 0) / dx
    # q[end] = -(1.0 - u[end]) / dx

    for i in 1:nx-1
        q[i+1] = -(u[i+1] - u[i]) / dx
    end

    for i in 2:nx-1
        r[i] = (q[i+1] - q[i]) / dx
    end

    return
end


nx = 5
dx = 1 / (nx - 1)

u = rand(nx)
r = zeros(nx)
q = zeros(nx + 1)

dr = zeros(nx)
dq = zeros(nx + 1)

J = zeros(nx, nx)

onehot = zeros(nx)

for irow in 1:nx
    onehot .= 0.0
    onehot[irow] = 1.0

    dr .= 0.0
    dq .= 0.0

    autodiff(Forward, residual!,
        Duplicated(r, dr),
        Duplicated(u, onehot),
        Duplicated(q, dq),
        Const(dx))

    J[irow, :] .= dr
end


∇u = zeros(nx - 1)

ū  = make_zero(u)
∇ū = make_zero(∇u)

autodiff(Reverse, E, Duplicated(u, ū), Duplicated(∇u, ∇ū), Const(dx))

J = J[2:end-1, 2:end-1]

@assert issymmetric(J)
@assert isposdef(J)

display(J)
display(r)
display(ū)

@show maximum(abs.(r .- ū))
