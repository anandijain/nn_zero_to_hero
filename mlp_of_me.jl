using Flux

function ix_maps(chars)
    stoi_ps = chars .=> eachindex(chars)
    stoi = Dict(stoi_ps)
    itos = Dict(reverse.(collect(stoi)))
    stoi, itos
end

t = filter(isascii, read("C:/Users/anand/src/transcript_grabber/dataset.txt", String))
l = length(t)
block_size = 3
split_idx = round(Int, 0.9 * l)

cs = sort(unique(t))
# semi is pad char
';' ∈ cs
chars = [';', cs...]
nc = length(chars)
stoi, itos = ix_maps(chars)
# join(getd(itos, getd(stoi, collect(t)))) == t

function str_to_ixs(ixs)
    getd(stoi, collect(t[ixs.+1]))
end
B = batch_size = 32
batch_inds() = rand(1:(split_idx-block_size), batch_size)
xixs = rand(1:(split_idx-block_size), batch_size)
function get_batch(ixs)
    Xb = stack(map(ix -> getd(stoi, collect(t[ix:ix+block_size-1])), ixs))
    Yb = str_to_ixs(ixs)
    (Xb, Yb)
end
Xb, Yb = get_batch(ixs)
get_n_batches(n) = map(x->get_batch(x), [batch_inds() for _ in 1:n])
(X, Y) = get_batch(collect(1:(l-block_size)))
Xb
Yb

T = block_size
C = n_embd = 2
nh = 100
emb = Embedding(nc, n_embd)
l1 = Dense(n_embd*block_size, nh, tanh)
l2 = Dense(nh, nc)

model = Chain(
    emb,
    x -> reshape(x, (n_embd * block_size, :)),
    l1,
    l2
)

out = model(X)
ee = emb(X)
ree = reshape(ee, (n_embd * block_size, :))
l2(l1(ree))

e = emb(Xb) # (C,T,B) (2, 3, 32)
l1()
l1.weight # (100 x 2 ) (nh x C )
# 100 x 2 * (2 x 3 x 32 )
# 100 x 3 x 32
la = l1(e) # (100)
lb = l2(la)

logits = model(Xb)
logits[:, :, 1]

loss = logitcrossentropy(model(X), onehotbatch(Y, 1:nc))


# batch_size = 32
Xtr_rows = eachrow(Xtr)
loader = Flux.DataLoader((axes(X, 2), Y), batchsize=batch_size, shuffle=true);
(x1, ybatch) = first(loader)
xbatch = X[:, x1]
model(@view(X[:, x1]))
loss, grads = Flux.withgradient(model) do m
    logits = m(@view(X[:, x1]))
    loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:nc))
end

losses = []
opt_state = Flux.setup(Flux.Descent(), model)  # will store optimiser momentum, etc.
nepochs = 1
for j in 1:nepochs
    for (i, (xbatch, ybatch)) in enumerate(loader)
        loss, grads = Flux.withgradient(model) do m
            logits = m(@view(X[:, xbatch]))
            loss = logitcrossentropy(logits, onehotbatch(ybatch, 1:nc))
        end

        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
    # loss_tr = logitcrossentropy(model(Xtr), onehotbatch(Ytr, 1:nc))
    # loss_dev = logitcrossentropy(model(Xdev), onehotbatch(Ydev, 1:nc))
    # @show (loss_tr, loss_dev)
    if j == 2
        opt_state = Flux.setup(Flux.Descent(0.01), model)
    elseif j == 3
        opt_state = Flux.setup(Flux.Descent(0.01), model)
    end
end

loss = logitcrossentropy(model(X), onehotbatch(Y, 1:nc))

function generate(model, n, block_size; maxlen=100)
    outs = []
    for _ in 1:n
        # xenc = ones(Int, block_size)
        xenc = rand(1:nc, block_size)
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

generate(model, 5, block_size)