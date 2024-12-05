using Flux, OneHotArrays, ProgressMeter, StatsBase
using Flux.Losses

function get_dataset(block_size)
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

words = readlines("names.txt")
achars = string.(sort(unique(join(words))))
chars = [".", achars...]
vocab_size = length(chars)
block_size = 3


stoi_ps = chars .=> eachindex(chars)
stoi = Dict(stoi_ps)
itos = Dict(reverse.(collect(stoi)))

X, Y = get_dataset(block_size)
nx = size(X, 1)

lr = 1
emb_dim = 2
h_dim = 100
batch_size = 32
# C = randn(vocab_size, emb_dim)
# W1 = randn(emb_dim * block_size, h_dim)
# b1 = randn(1, h_dim)
# W2 = randn(h_dim, vocab_size)
# b2 = randn(1, vocab_size)
# ps = Params([C, W1, b1, W2, b2])


C = Embedding(vocab_size, emb_dim)
mlp = Chain(
    Dense(block_size * emb_dim, h_dim, tanh),
    Dense(h_dim, vocab_size)
)

model = Chain(
    Embedding(vocab_size, emb_dim),
    x -> reshape(permutedims(x, (2, 1, 3)), (:, block_size * emb_dim))',
    Dense(block_size * emb_dim, h_dim, tanh),
    Dense(h_dim, vocab_size)
)

C([1, 1, 1]) # get the embedding for a single example returns a 2x3 
C(X)

# fwd pass
emb = reshape(permutedims(C(X), (2, 1, 3)), (:, block_size * emb_dim))'
logits = mlp(emb)
loss = logitcrossentropy(logits, onehotbatch(Y, 1:27))

loader = Flux.DataLoader((eachrow(X), Y), batchsize=64, shuffle=true);
losses = []
opt_state = Flux.setup(Flux.Descent(0.01), model)  # will store optimiser momentum, etc.

(xbatch, ybatch) = first(loader)
xin = stack(xbatch; dims=1)
logits = model(xin)
counts = exp.(logits[:, 1] .- maximum(logits[:, 1]))
sample(Weights(counts))
loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:27))
loss, grads = Flux.withgradient(model) do m
    logits = m(stack(xbatch; dims=1))
    loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:27))
end
Flux.update!(opt_state, model, grads[1])

@showprogress for epoch in 1:1
    for (xbatch, ybatch) in loader
        loss, grads = Flux.withgradient(model) do m
            logits = m(stack(xbatch; dims=1))
            loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:27))
        end
        @show loss
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

for i in 1:5
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
    println(join(out))

end