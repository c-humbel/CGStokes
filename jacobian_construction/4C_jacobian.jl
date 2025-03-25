using KernelAbstractions
using Enzyme
using Random
using LinearAlgebra
using CUDA

include("../scripts/4_nonlinear_augmLagrange/kernels_free_slip.jl")
include("../scripts/4_nonlinear_augmLagrange/kernels_volume_fractions.jl")
include("../src/tuple_manip.jl")


n=5
seed=1234

rng       = MersenneTwister(seed)
backend   = CPU()
type      = Float64
workgroup = 64

Lx, Ly = 1, 1
nx, ny = n, n
rₐ = 1.6Lx
rₛ = 0.9Ly
x₀ = 0
y₀ = -1.2Ly

γ      = 5.0
q      = 1.33
ϵ̇_bg   = eps()

Δx,  Δy  = Lx / nx, Ly / ny
iΔx, iΔy = inv(Δx), inv(Δy)
xc = LinRange(-0.5Lx + 0.5Δx, 0.5Lx - 0.5Δx, nx)
yc = LinRange(-0.5Ly + 0.5Δy, 0.5Ly - 0.5Δy, ny)
xv = LinRange(-0.5Lx, 0.5Lx, nx+1)
yv = LinRange(-0.5Ly, 0.5Ly, ny+1)


P    = (c=KernelAbstractions.zeros(backend, type, nx, ny),
        v=KernelAbstractions.zeros(backend, type, nx+1, ny+1))
P₀   = deepcopy(P)  # old pressure
P̄    = deepcopy(P)  # memory needed for auto-differentiation
B    = deepcopy(P)  # prefactor of constituitive relation
τ    = (c=(xx=KernelAbstractions.zeros(backend, type, nx  , ny  ),
            yy=KernelAbstractions.zeros(backend, type, nx  , ny  ),
            xy=KernelAbstractions.zeros(backend, type, nx+2, ny+2)),
        v=(xx=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
            yy=KernelAbstractions.zeros(backend, type, nx+1, ny+1),
            xy=KernelAbstractions.zeros(backend, type, nx+1, ny+1)))  # deviatoric stress tensor
τ̄    = deepcopy(τ)
V    = (xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
        yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
        xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
        yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
V̄    = deepcopy(V)  # memory needed for auto-differentiation
D    = deepcopy(V)  # search direction of CG, cells affecting Dirichlet BC are zero
R    = deepcopy(V)  # nonlinear Residual
Q    = deepcopy(V)  # Jacobian of compute_R wrt. V, multiplied by some vector (used for autodiff)
f    = deepcopy(V)  # body force
ωₐ   = (c =KernelAbstractions.zeros(backend, type, nx+2, ny+2),
        v =KernelAbstractions.zeros(backend, type, nx+1, ny+1),
        xc=KernelAbstractions.zeros(backend, type, nx+1, ny),
        yc=KernelAbstractions.zeros(backend, type, nx, ny+1),
        xv=KernelAbstractions.zeros(backend, type, nx+2, ny+1),
        yv=KernelAbstractions.zeros(backend, type, nx+1, ny+2))
ωₛ   = deepcopy(ωₐ)

R̂    = deepcopy(V)
P̂    = deepcopy(P)
τ̂    = deepcopy(τ)

fill!.(values(B), 1)

for a = V
    copyto!(a, rand(rng, size(a)...))
end

copyto!(f.yc, rand(rng, size(f.yc)...))
copyto!(f.yv, rand(rng, size(f.yv)...))
fill!(f.xc, eps())
fill!(f.xv, eps())



# initialise_volume_fractions_ring_segment!(ωₐ, ωₛ, x₀, y₀, rₐ, rₛ, xc, yc, xv, yv)
tplFill!(ωₐ, 1.)
tplFill!(ωₛ, 1.)

comp_P_τ!  = compute_P_τ_weighted!(backend, workgroup, (nx+1, ny+1))
comp_R!    = compute_R_weighted!(backend, workgroup, (nx+2, ny+2))

function jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, V̄, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)
    tplSet!(P̂, P)
    tplSet!(τ̂.c, τ.c)
    tplSet!(τ̂.v, τ.v)
    make_zero!(Q)
    make_zero!(P̄)
    make_zero!(τ̄)
    autodiff(Forward, comp_P_τ!,
            DuplicatedNoNeed(P̂, P̄), DuplicatedNoNeed(τ̂, τ̄),
            Const(P₀),
            Duplicated(V, V̄),
            Const(B), Const(q), Const(ωₛ), Const(ϵ̇_bg), Const(iΔx), Const(iΔy), Const(γ))

    autodiff(Forward, comp_R!,
            DuplicatedNoNeed(R̂, Q),
            Duplicated(P, P̄), Duplicated(τ, τ̄),
            Const(f), Const(ωₐ), Const(ωₛ), Const(iΔx), Const(iΔy))
    return nothing
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
@views for d = D
    for I = eachindex(d)
        # set one entry in search vector to 1
        CUDA.@allowscalar d[I] = 1.0
        # compute the jacobian column by multiplying it with a "basis vector"
        jvp_R(R̂, Q, P, P̄, P̂, τ, τ̄, τ̂, V, D, P₀, f, B, q, ωₐ, ωₛ, ϵ̇_bg, iΔx, iΔy, γ)

        # store result in jacobian
        J[1:E_xc, col]      .= reshape(Q.xc, N_xc)
        J[E_xc+1:E_yc, col] .= reshape(Q.yc, N_yc)
        J[E_yc+1:E_xv, col] .= reshape(Q.xv, N_xv)
        J[E_xv+1:end , col]  .= reshape(Q.yv, N_yv)
        # increase column count
        col += 1
        # reset search vector
        tplFill!(D, 0.)
    end
end

