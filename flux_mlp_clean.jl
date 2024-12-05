using Flux, OneHotArrays, ProgressMeter, StatsBase, Plots, Random
using Flux.Losses

function get_dataset(words, block_size)
    X, Y = [], Int[]
    for w in words
        context = ones(Int, block_size)
        for c in w * "."
            c = string(c)
            ix = stoi[c]
            push!(X, context)
            push!(Y, ix)
            # println(join(getd(itos, context)), "-->", itos[ix])
            context = cat(context[2:end], [ix]; dims=1)
            # break
        end
        # break
    end

    X = stack(X, dims=1)
    Y
    (X, Y)
end

function generate(model, n)
    outs = []
    for i in 1:n
        xenc = ones(Int, block_size)
        out = []
        while true
            logits = model(reshape(xenc, (1, 3)))
            counts = exp.(logits[:, 1] .- maximum(logits[:, 1]))
            ix = sample(Weights(counts))
            push!(out, itos[ix])
            if ix == 1
                break
            end
            circshift!(xenc, -1)
            xenc[3] = ix
        end
        o = join(out)
        println(o)
        push!(outs, o)
    end
    outs 
end


words = readlines("names.txt")
nw = length(words)
achars = string.(sort(unique(join(words))))
chars = [".", achars...]
vocab_size = length(chars)
block_size = 3


stoi_ps = chars .=> eachindex(chars)
stoi = Dict(stoi_ps)
itos = Dict(reverse.(collect(stoi)))

lr = 1
emb_dim = 2
h_dim = 200
batch_size = 32

model = Chain(
    Embedding(vocab_size, emb_dim),
    x -> reshape(permutedims(x, (2, 1, 3)), (:, block_size * emb_dim))',
    Dense(block_size * emb_dim, h_dim, tanh; init=randn32),
    Dense(h_dim, vocab_size; init=randn32)
)

shuffled_words = shuffle(words)
Xs, Ys = get_dataset(shuffled_words, block_size)
tr_idx = round(Int, nw * 0.8)
dev_idx = round(Int, nw * 0.9)
Xtr, Ytr = get_dataset(shuffled_words[1:tr_idx], block_size)
Xdev, Ydev = get_dataset(shuffled_words[(tr_idx+1):dev_idx], block_size)
Xte, Yte = get_dataset(shuffled_words[(dev_idx+1):end], block_size)

@test reduce(vcat, [Xtr, Xdev, Xte]) == Xs

loader = Flux.DataLoader((eachrow(Xtr), Ytr), batchsize=32, shuffle=true);
losses = []
opt_state = Flux.setup(Flux.Descent(0.01), model)  # will store optimiser momentum, etc.

for (i, (xbatch, ybatch)) in enumerate(loader)
    loss, grads = Flux.withgradient(model) do m
        logits = m(stack(xbatch; dims=1))
        loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:27))
    end
    Flux.update!(opt_state, model, grads[1])
    push!(losses, loss)  # logging, outside gradient context
end

loss = logitcrossentropy(model(Xtr), onehotbatch(Ytr, 1:27))
loss = logitcrossentropy(model(Xdev), onehotbatch(Ydev, 1:27))

generate(model, 5)

# visualize embedding
emb_xys = eachcol(model.layers[1].weight)
uxys = unzip(emb_xys)
pl = scatter(uxys)
for (i, emb_xy) in enumerate(emb_xys)
    annotate!(emb_xy..., chars[i])
end
pl