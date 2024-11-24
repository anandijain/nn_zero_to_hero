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