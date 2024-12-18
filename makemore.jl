using Plots, Distributions, Random, StatsBase, LinearAlgebra
using Zygote

w_chars(w) = [".", string.(collect(w))..., "."]
zipit(wcs) = zip(wcs, wcs[2:end])

function build_bigram_heatmap_array(ws)
    N = zeros(Int32, 27, 27)
    for w in ws
        wcs = w_chars(w)
        for (a, b) in zipit(wcs)
            N[stoi[a], stoi[b]] += 1
        end
    end
    N
end

function onehot(vec, n)
    ys = []
    for x in vec
        y = zeros(n)
        y[x] = 1
        push!(ys, y)
    end
    ys
end
function norm_rows(A)
    rs = eachrow(A)
    stack(rs ./ sum.(rs), dims=1)
end

ws = readlines("names.txt")
cs = string.(sort(unique(join(ws))))

extrema(length.(ws))

stoi = Dict(cs .=> (eachindex(cs) .+ 1))
stoi["."] = 1
alpha = [".", cs...] 
itos = Dict(reverse.(collect(stoi)))

T = tally(reduce(vcat, collect.(zipit.(w_chars.(ws)))))

N = build_bigram_heatmap_array(ws)
# smoothing 
N .+= 1
N[1, :]
p = N[1, :] ./ sum(N[1, :])

sum(p)
seed = rand(UInt)
rng = MersenneTwister(seed)

dist = Categorical(p)
ixs = rand(rng, dist, 10000)
getd(itos, ixs) |> tally

pvecs = map(x -> normalize(x, 1), eachrow(N))
P = collect(reduce(hcat, pvecs)')
@assert sum(pvecs[1]) ≈ 1.0
@assert allapprox(sum.(pvecs))
dists = map(Categorical, pvecs)

for i in 1:30
    out = []
    ix = 1
    while true
        ix = only(rand(rng, dists[ix], 1))
        push!(out, itos[ix])
        if ix == 1
            break
        end
    end
    println(join(out))
end


#nll
log_likelihood = 0.0
n = 0

# shows anand is a relatively common name 
# for w in ["anand"]
for w in ws
    wcs = w_chars(w)
    for (a, b) in zipit(wcs)
        ix1, ix2 = getd(stoi, (a, b))
        prob = P[ix1, ix2]
        logprob = log(prob)
        log_likelihood += logprob
        n += 1
        # println("$(a)$(b): $(prob) $logprob")
    end
end
@show log_likelihood
nll = -log_likelihood
@show nll
# its typical to have the loss be the average nll 
@show nll / n


# train set
xs, ys = [], []

for w in ws
    wcs = w_chars(w)
    for (a, b) in zipit(wcs)
        ix1, ix2 = getd(stoi, (a, b))
        push!(xs, ix1)
        push!(ys, ix2)
    end
end

getd(itos, xs)
getd(itos, ys)
# heatmap(stack(onehot(xs[1:5], 27))')
xenc = stack(onehot(xs, 27))'
num = size(xenc, 1)
W = randn(27, 27)
logits = xenc * W # "log counts"
counts = exp.(logits) # analogous to count mat N above 
probs = norm_rows(counts)

gt_preds = probs[CartesianIndex.(1:num, ys)] # this is the predicted probabilities for the correct answer 
# so for xs[1] "." its saying that 'e' comes next predicted .05 probability

# i just dont understand how the loss is not done via comparison to something 
loss = -mean(log.(gt_preds))


λ=0.01
for i in 1:100
    g =
        gradient(Params([W])) do
            logits = xenc * W # "log counts"
            counts = exp.(logits) # analogous to count mat N above 
            probs = counts ./ sum(counts, dims=2)
            nums = probs[CartesianIndex.(2:num, ys[2:end])]
            loss = -mean(log.(nums)) #+ λ * sum(W .^ 2)
            @show i, loss
            loss
        end

    dl_dW = g[W]

    W += -50 * dl_dW

end
# generation
for i in 1:5
    out = []
    ix = 1
    while true
        xenc_ = stack(onehot([ix], 27))'
        logits = xenc_ * W
        counts = exp.(logits)
        p = norm_rows(counts)
        # @show p
        d = Categorical(vec(p))
        ix = rand(d)
        # @show ix
        push!(out, itos[ix])
        if ix == 1
            break
        end

    end
    println(join(out))
end

