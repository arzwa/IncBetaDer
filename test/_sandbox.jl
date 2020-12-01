# ---------------------------------------------------------------------------
# ChainRules, however ForwardDiff does not use ChainRules, but only
# DiffRules...
# using ChainRules, ChainRulesCore
# 
# function frule((_, Δp, Δq, Δx), ::typeof(SpecialFunctions.beta_inc), p, q, x)
#     z, ∂p, ∂q, ∂x = beta_inc_grad(p, q, x)
#     Ω = (z, 1. - z)  # beta_inc returns a tuple (z, 1. z) (not sure why)
#     ∂Ω1 = (∂p, ∂q, ∂x)
#     ∂Ω2 = (-∂p, -∂q, -∂x) 
#     ∂Ω = Composite{typeof(Ω)}(∂Ω1, ∂Ω2)  # is this correct wrapping of tuples?
#     return (Ω, ∂Ω)
# end
# 
# p, q, x = 0.2, 0.3, 0.1
# primal = beta_inc(p, q, x)
# g1 = grad(central_fdm(5,1), y->beta_inc(y...)[1], [p, q, x])[1]
# g2 = grad(central_fdm(5,1), y->beta_inc(y...)[2], [p, q, x])[1]
# Ω, ∂Ω = frule((Zero(), 1., 1., 1.), beta_inc, p, q, x)
# @info "results" primal Ω g1 g2 ∂Ω[1] ∂Ω[2]

# But DiffRules is not really designed for more complicated cases...
# https://github.com/JuliaDiff/ForwardDiff.jl/issues/413

p, q, x = 2., 3., 0.2
@time g = grad(central_fdm(5,1), y->beta_inc_inv(y...)[1], [p, q, x])[1]
@time ForwardDiff.gradient(y->beta_inc_inv(y...), [p, q, x])

p, q, x = 0.2, 0.3, 0.1
@time g = grad(central_fdm(5,1), y->beta_inc_inv(y...)[1], [p, q, x])[1]
@time ForwardDiff.gradient(y->beta_inc_inv(y...), [p, q, x])

# It seems getting ForwardDiff to work with ChainRules is WIP
# (ForwardDiff2...)
# ----------------------------------------------------------------------------


ForwardDiff.gradient(x->beta_inc_inv(x...), [2., 3., 0.1])

# ERROR: MethodError: no method matching
# beta_inc_inv(::Dual{ForwardDiff.Tag{var"#72#73",Float64},Float64,2},
#              ::Dual{ForwardDiff.Tag{var"#72#73",Float64},Float64,2}, 
#              ::Float64)

# We need a less specific beta_inc_inv method

# It seems to work for most values of α, β, but fails if these get too small,
# with exploding gradients
d = Beta(20,30)
data = map(1:100) do _
    p = rand(d)
    rand(Geometric(p), 100)
end

# This takes the median of each 1/K portion of the probability mass
function discretize_betaq(a, b, K)
    qstart = 1.0/(2.0 * K)
    qend = 1. - qstart
    xs = [beta_inc_inv_(a, b, x) for x in qstart:(1/K):qend]
    xs .* ((K*a/(a + b))/sum(xs))
end

# This obtains the means for each 1/K portion of the probability mass
# This requires quite a bit more `beta_inc_` calls...
function discretize_betam(a::T, b::T, K) where T
    @assert a > one(a) && b > one(b) "only defined for a,b > 1"
    step = 1.0/K
    qs = [beta_inc_inv_(a, b, x) for x in 0.0:step:1.0]
    bab  = beta(a, b)
    bapb = beta(a+1, b)
    xs = zeros(T, K)
    for i=2:K+1
        N = beta_inc_(a+1, b, qs[i]) - beta_inc_(a+1, b, qs[i-1])
        xs[i-1] = bapb*N/bab
    end
    # how to properly normalize in the loop?
    return xs .* (a / (mean(xs) * (a + b)))
end

using StatsFuns

@model turingtest(data, K, method=1) = begin
    a ~ Uniform(1.1, 100.)
    b ~ Uniform(1.1, 100.)
    q = method == 1 ? discretize_betaq(a, b, K) : discretize_betam(a, b, K)
    any(q .< zero(q[1])) || any(q .> one(q[1])) && return -Inf
    for i=1:length(data)
        Turing.@addlogprob! -log(K) + 
            logsumexp([loglikelihood(Geometric(q_), data[i]) for q_ in q])
    end
end

chain1 = sample(turingtest(data, 4, 1), NUTS(), 1000)

chain2 = sample(turingtest(data, 4, 2), NUTS(), 1000)
