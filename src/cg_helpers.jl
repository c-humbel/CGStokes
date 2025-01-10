using Enzyme

function compute_α(R, Q, P, P̄, P₀, V, V̄, D, ρg, η, μ, dx, dy, γ)
    # compute Jacobian-vector product Jac(R) * D using Enzyme
    # result is stored in Q
    tplSet!(V̄, D) # need to copy D since autodiff may change it
    autodiff(Forward, compute_R!, DuplicatedNoNeed(R, Q),
             Duplicated(P, P̄), Const(P₀), Duplicated(V, V̄),
             Const(ρg), Const(η), Const(dx), Const(dy), Const(γ))
    # compute α = dot(R, M*R) / dot(D, A*D)
    # -> since R = rhs - A*V, ∂R/∂V * D = -A * D
    #    therefore we use here the negative of the Jacobian-vector product
    return  μ / tplDot(D, Q, -1.)
end