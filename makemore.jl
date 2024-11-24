using Plots, Distributions, Random
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
ws = readlines("names.txt")
cs = string.(sort(unique(join(ws))))

extrema(length.(ws))

stoi = Dict(cs .=> (eachindex(cs) .+ 1))
stoi["."] = 1

itos = Dict(reverse.(collect(stoi)))

tally(reduce(vcat, collect.(zipit.(w_chars.(ws)))))

N = build_bigram_heatmap_array(ws)
N[1, :]
p = N[1, :] ./ sum(N[1, :])
sum(p)
seed = rand(UInt)
rng = MersenneTwister(seed)

dist = Categorical(p)
ixs = rand(rng, dist, 100)
getd(itos, ixs) |> tally

pvecs = map(x -> normalize(x, 1), eachrow(N))
P = collect(reduce(hcat, pvecs)' )
@assert allapprox(sum.(pvecs))
dists = map(Categorical, pvecs)

for i in 1:30
    out = []
    ix = 1
    while true
        ix = only(rand(rng, dists[ix], 1))
        push!(out, itos[ix])
        # print(itos[ix])
        if ix == 1
            break
        end
    end
    println(join(out))
end


#nll
log_likelihood = 0.
n = 0

for w in ws 
    wcs = w_chars(w)
    for (a, b) in zipit(wcs)
        ix1, ix2 = getd(stoi, (a, b))
        prob = P[ix1, ix2]
        logprob = log(prob)
        log_likelihood += logprob
        n +=1 
    end
end
@show log_likelihood
nll = -log_likelihood
@show nll
@show nll/n


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

function onehot(vec, n)  
    ys = []
    for x in vec 
        y = zeros(n)
        y[x] = 1
        push!(ys, y)
    end
    ys 
end
    
heatmap(stack(onehot(xs[1:5], 27))')