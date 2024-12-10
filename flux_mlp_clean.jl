using Flux, OneHotArrays, ProgressMeter, StatsBase, Plots, Random, BenchmarkTools, LinearAlgebra
using Graphs
using Flux.Losses
using Flux: glorot_uniform, kaiming_normal

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

function generate(model, n, block_size; maxlen=100)
    outs = []
    for _ in 1:n
        xenc = ones(Int, block_size)
        out = []
        i = 1
        while true
            logits = model(reshape(xenc, (1, block_size)))
            counts = exp.(logits[:, 1] .- maximum(logits[:, 1]))
            ix = sample(Weights(counts))
            push!(out, itos[ix])
            if ix == 1 || i == maxlen
                break
            end
            circshift!(xenc, -1)
            xenc[3] = ix
            i += 1
        end
        o = join(out)
        # println(o)
        push!(outs, o)
    end
    outs
end

# visualize embedding
function embedding_plot(model)
    C = model.layers[1].weight

    emb_xys = eachcol(C)
    xlims = extrema(C[1, :])
    ylims = extrema(C[1, :])
    uxys = unzip(emb_xys)

    pl = scatter()
    zuxt = collect(zip(uxys..., chars))
    annotate!(pl, zuxt)
    xlims!(pl, xlims)
    ylims!(pl, ylims)
    pl
end


words = readlines("names.txt")
nw = length(words)
achars = string.(sort(unique(join(words))))
chars = [".", achars...]
vocab_size = length(chars)

stoi_ps = chars .=> eachindex(chars)
stoi = Dict(stoi_ps)
itos = Dict(reverse.(collect(stoi)))

block_size = 3

shuffled_words = shuffle(words)
Xs, Ys = get_dataset(shuffled_words, block_size)
tr_idx = round(Int, nw * 0.8)
dev_idx = round(Int, nw * 0.9)
Xtr, Ytr = get_dataset(shuffled_words[1:tr_idx], block_size)
Xdev, Ydev = get_dataset(shuffled_words[(tr_idx+1):dev_idx], block_size)
Xte, Yte = get_dataset(shuffled_words[(dev_idx+1):end], block_size)

emb_dim = 10
n_hidden = 200

# model
embb = emb = Embedding(vocab_size, emb_dim) #;init=zeros),
reshape_permute = x -> reshape(permutedims(x, (2, 1, 3)), (:, block_size * emb_dim))'
l1 = Dense(block_size * emb_dim, n_hidden, relu; init=kaiming_normal)
l2 = Dense(n_hidden, vocab_size; init=kaiming_normal)

model = Chain(
    emb, 
    reshape_permute,
    l1,
    l2
)

# embedding_plot(model)
# taking a batch
x = Xs[1:32, :]
ee = emb(x)
model(x)
# x[1,:], Ys[1]
hf = ∘(reverse(model.layers[1:3])...)
hf = model[1:3]
hf(x)
hpf = x -> model.layers[3].weight * ∘(reverse(model.layers[1:2])...)(x) .+ l1.bias
h = hf(x)
histogram(vec(h); bins=50)
heatmap(abs.(h) .> 0.9)


# note that dense layers init bias to 0 
# model.layers[4].weight .*= 0.01
loss_tr = logitcrossentropy(model(Xtr), onehotbatch(Ytr, 1:27))

# @test reduce(vcat, [Xtr, Xdev, Xte]) == Xs


batch_size = 4
Xtr_rows = eachrow(Xtr)
loader = Flux.DataLoader((axes(Xtr, 1), Ytr), batchsize=batch_size, shuffle=true);
(x1, y1) = first(loader)
batch = Xtr[x1, :]
losses = []
opt = Flux.Adam(0.01)
opt_state = Flux.setup(Flux.Descent(), model)  # will store optimiser momentum, etc.
nepochs=3
for j in 1:nepochs
    for (i, (xbatch, ybatch)) in enumerate(loader)
        loss, grads = Flux.withgradient(model) do m
            logits = m(@view(Xtr[xbatch, :]))
            loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:27))
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
    loss_tr = logitcrossentropy(model(Xtr), onehotbatch(Ytr, 1:27))
    loss_dev = logitcrossentropy(model(Xdev), onehotbatch(Ydev, 1:27))
    @show (loss_tr, loss_dev)
    if j == 2
        opt_state = Flux.setup(Flux.Descent(0.01), model)
    elseif j == 3
        opt_state = Flux.setup(Flux.Descent(0.01), model)
    end
end

plot(log10.(losses))

loss_tr = logitcrossentropy(model(Xtr), onehotbatch(Ytr, 1:27))
loss_dev = logitcrossentropy(model(Xdev), onehotbatch(Ydev, 1:27))


# embedding_plot(model)

h = hf(Xs[1:200, :])

# this plot doesn't make sense for non-tanh 
heatmap(abs.(h) .> .99)
heatmap(h .== 0)
histogram(vec(h); bins=50)
histogram(log.(vec(h)); bins=50)

generate(model, 5, block_size)

hp = hpf(x)
u = mean(hp;dims=2)
s = std(hp;dims=2)
bngain = ones(n_hidden, 1)
bnbias = zeros(n_hidden, 1)
normed_hp = (hp .- u) ./ s
bn_hp = bngain .* normed_hp .+ bnbias

