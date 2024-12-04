# https://www.youtube.com/watch?v=TCH_1BHY58I
using Distributions, Random, StatsBase, LinearAlgebra
using Zygote, NNlib, Flux, Flux.Losses
using OneHotArrays

function onehot(vec, n)
    ys = zeros(length(vec), n)
    for (i, x) in enumerate(vec)
        ys[i, x] = 1
    end
    ys
end
function norm_rows!(A)
    A ./= sum(A; dims=2)
end

function norm_rows(A)
    A ./ sum(A; dims=2)
end

words = readlines("names.txt")
achars = string.(sort(unique(join(words))))
chars = [".", achars...]
vocab_size = length(chars)


stoi_ps = chars .=> eachindex(chars)
stoi = Dict(stoi_ps)
# stoi["."] = 1
itos = Dict(reverse.(collect(stoi)))

block_size = 3
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
nx = size(X, 1)

C = randn(27, 2)
@test C[5, :] ≈ vec(onehot([5], 27) * C)

foo = C[X] # C[X] in torch is 32x3x2
C

getd(itos, X[1, :])

emb = reshape(C[vec(X), :], (size(X, 1), 3, 2))

emb[1, :, :]

# 32x2x3 
embb = permutedims(emb, (1, 3, 2))

W1 = randn(6, 100) # 6 because 3 sets of dim 2 embedding
b1 = randn(1, 100)
@test_throws Any emb * W1

# first emb
# embeddings for the first character and the first 10 examples of the train set 
show(stdout, "text/plain", emb[1:10, 1, :])


# emb.view(-1, 6)
input = reshape(emb, (size(X, 1), 6)) # WRONG! look at input[1, :]
reshape(emb, (:, 6))
#=
since the first X,Y pair  = "..." --> "e" for the start of emma
we expect the first row of the input to be 3 copies of the embedding for "."
    but thats not what we get
emb[1, :, :] -> 
emb[1, 1, :]


vec(emb[1, :, :]')

=#
# @view emb[32, -1]

# this seems to work! 
embr = reshape(embb, (:, 6))
h = tanh.(embr * W1 .+ b1)


W2 = randn(100, vocab_size)
b2 = randn(1, 27)
logits = (h * W2) .+ b2
counts = exp.(logits)
probabilities = norm_rows!(counts)
# "..." -> "e"
loss = -mean(log.(probabilities[CartesianIndex.(1:length(Y), Y)]))



# cleaned up version
lr = .0001
emb_dim = 2
batch_size = 10000
h_dim = 100
C = randn(vocab_size, emb_dim)
W1 = randn(emb_dim * block_size, h_dim)
b1 = randn(1, h_dim)
W2 = randn(h_dim, vocab_size)
b2 = randn(1, vocab_size)
ps = Params([C, W1, b1, W2, b2])

# fwd pass, doesn't work in zygote
emb = reshape(permutedims(reshape(C[vec(X), :], (:, block_size, emb_dim)), (1, 3, 2)), (:, 6))
logits = tanh.(emb * W1 .+ b1) * W2 .+ b2
println(logitcrossentropy(logits, onehot(Y, 27); dims=2))

#non mutating fwd pass
# ps = Params([C, W1, b1, W2, b2])

# TODO need to see if logitcrossentropy improves the loss
# currently with minibatching and without getting a min loss of 4.5ish 
# compared to andrej with 2.7 at this stage
for i in 1:100
    idxs = rand(1:nx, batch_size)
    batch = X[idxs, :]
    y_batch = Y[idxs]
    g = gradient(Params([C, W1, b1, W2, b2])) do
        emb = reshape(permutedims(reshape(C[vec(batch), :], (:, block_size, emb_dim)), (1, 3, 2)), (:, 6))
        logits = tanh.(emb * W1 .+ b1) * W2 .+ b2
        counts = exp.(logits)
        probabilities = norm_rows(counts)
        loss = -mean(log.(probabilities[CartesianIndex.(1:batch_size, y_batch)]))
        @show loss
        loss
    end
    C += -lr * g[C]
    W1 += -lr * g[W1]
    b1 += -lr * g[b1]
    W2 += -lr * g[W2]
    b2 += -lr * g[b2]
end


# generation
for i in 1:5
    out = []
    # ix = 1
    xenc = ones(Int, 3)
    while true
        emb = reshape(permutedims(reshape(C[vec(xenc), :], (:, block_size, emb_dim)), (1, 3, 2)), (:, 6))
        logits = tanh.(emb * W1 .+ b1) * W2 .+ b2
        counts = exp.(logits)
        probabilities = norm_rows(counts)
        # @show p
        d = Categorical(vec(probabilities))
        ix = rand(d)
        # @show ix
        l = itos[ix]
        push!(out, l)
        if ix == 1
            break
        end
        push!(xenc, ix)
        xenc = xenc[2:end]

    end
    println(join(out))
end

# check whats wrong when generating with random weights, too many V for my likey
xenc = ones(Int, block_size)
emb = reshape(permutedims(reshape(C[vec(xenc), :], (:, block_size, emb_dim)), (1, 3, 2)), (:, 6))
logits = tanh.(emb * W1 .+ b1) * W2 .+ b2
counts = exp.(logits)
probabilities = norm_rows(counts)
# @show p
d = Categorical(vec(probabilities))
ix = rand(d)
l = itos[ix]


