include("Ismip_E000_Arolla.jl")

args = Dict(:n              => 10,
            :aspect         => 0.3,
            :γ_factor       => 1e5,
            :niter          => 5000n,
            :ϵ_ph           => 1e-8,
            :ϵ_cg           => 1e-10,
            :ϵ_newton       => 1e-9,
            :freq_recompute => 50,
            :verbose        =>false)

out = run("data/arolla100.dat"; args...)
dat = extract_data(out...)
create_summary_plots(dat...; savefig=false)
