include("manufactured_solution.jl")

args = Dict(:n              => 1000,
            :γ_factor       => 1e5,
            :niter          => 1e7,
            :freq_recompute => 50,
            :ϵ_ph           => 1e-8,
            :ϵ_newton       => 1e-8,
            :ϵ_cg           => 1e-10,
            :backend        => CUDABackend(),
            :workgroup      => (32, 8),
            :verbose        => true,
            :save           => true)

out = run_manufactured_solution(;args...)
