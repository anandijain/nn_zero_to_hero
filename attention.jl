using Flux, OneHotArrays, ProgressMeter, StatsBase, Plots, Random, BenchmarkTools, LinearAlgebra
using Graphs
using Flux.Losses
using Flux: glorot_uniform, kaiming_normal

bit_tril(H, W) = tril(trues(H, W))
function wei_tril(C)
    C ./ axes(C, 1)
end

words = readlines("names.txt")
nw = length(words)
achars = string.(sort(unique(join(words))))
chars = [".", achars...]
vocab_size = length(chars)

stoi_ps = chars .=> eachindex(chars)
stoi = Dict(stoi_ps)
itos = Dict(reverse.(collect(stoi)))


# https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#Attention
B,T,C = (4, 3, 2)
X = rand(1:10, B, T, C)

# embedding of the first character 
X[1, :, :]
X[1, 1, :]
X[1, 2, :]

# in this scenario though, all 8 tokens are used to predict the 9th IIUC
# so i dont see why we need to mask out future tokens, shouldn't they 
# be allowed to talk to eachother 

a = tril(ones(Bool, 4, 4))
b = tril(trues(4, 4))
sizeof(a)
sizeof(b)
sizeof(true)

A = tril(trues(3, 3))
B = rand(1:10, (3, 2))

Cm = A * B
1:(size(Cm, 1))
@btime axes(Cm, 1)
@btime Cm ./ axes(Cm, 1)
res = Cm ./ axes(Cm, 1)

# julia> X
# 4×8×2 Array{Float64, 3}:
wei = wei_tril(bit_tril(T, T))
# 8×8 Matrix{Float64}:
# xp = permutedims(X, (2, 3, 1))


# batched mul works when the batch is the last dimension
Xb = permutedims(X, (2,3,1))
# todo  figure out NNlib.batched_mul
Cm = wei ⊠ Xb
# Cm = X ⊠ wei
Cm[: ,:, 1]
# (T x T) * (B x T x C) --> (B T C)
result = stack([wei * X[b, :, :] for b in axes(X, 1)];dims=1) #3.614 μs (30 allocations: 83.39 KiB)
@btime stack([wei * X[b, :, :] for b in axes(X, 1)];dims=1)

@btime wei ⊠ Xb
# figure out how to make batched_mul! work too, for speed 
wei ⊠ Xb
result[1, :, :]

# is this identical to C 
n_embd = 2 
block_size = 3
token_embl = Embedding(vocab_size, n_embd)
pos_embl = Embedding(block_size, n_embd)
lm_head = Dense(n_embd, vocab_size)

# (B, T)
batch
tok_emb = token_embl(batch) # (C, B, T)

# reminder that Embedding is literally just a performant lookup 
# token_embl.(batch)

pos_embl = pos_embl(Base.OneTo(T)) # (nembd x T)

tok_emb

M = rand(5, 10)

#column major stuff
# M = rand(5, 10)
# M[2]
# M = rand(5, 10,2 )
# M[51]