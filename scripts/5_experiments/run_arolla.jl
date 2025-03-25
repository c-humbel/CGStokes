include("Ismip.jl")

args = Dict(:n              => 1000,
            :aspect         => 0.3,
            :γ_factor       => 1e5,
            :niter          => 5000000,
            :ϵ_ph           => 1e-8,
            :ϵ_cg           => 1e-10,
            :ϵ_newton       => 1e-8,
            :freq_recompute => 50,
            :verbose        => true,
            :backend        => CUDABackend(),
            :workgroup      => (32, 8))

out = run("data/arolla100.dat"; args...)
dat = extract_data(out...)
create_summary_plots(dat...; savefig=true)
