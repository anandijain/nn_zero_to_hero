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
h_dim = 200

model = Chain(
    Embedding(vocab_size, emb_dim),
    x -> reshape(permutedims(x, (2, 1, 3)), (:, block_size * emb_dim))',
    Dense(block_size * emb_dim, h_dim, tanh; init=randn32),
    Dense(h_dim, vocab_size; init=randn32)
)


# @test reduce(vcat, [Xtr, Xdev, Xte]) == Xs


batch_size = 1000
loader = Flux.DataLoader((eachrow(Xtr), Ytr), batchsize=batch_size, shuffle=true);
losses = []
opt_state = Flux.setup(Flux.Descent(0.1), model)  # will store optimiser momentum, etc.
nepochs=5
for j in 1:nepochs
    for (i, (xbatch, ybatch)) in enumerate(loader)
        loss, grads = Flux.withgradient(model) do m
            logits = m(stack(xbatch; dims=1))
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
        opt_state = Flux.setup(Flux.Descent(0.001), model)
    end
end

generate(model, 5, block_size)
plot(log10.(losses))

# embedding_plot(model)
