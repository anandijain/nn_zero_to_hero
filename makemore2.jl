# https://www.youtube.com/watch?v=TCH_1BHY58I

function onehot(vec, n)
    ys = zeros(length(vec), n)
    for (i, x) in enumerate(vec)
        ys[i, x] = 1
    end
    ys
end

words = readlines("names.txt")
achars = string.(sort(unique(join(ws))))
chars = [".", cs...]

stoi = Dict(cs .=> (eachindex(cs) .+ 1))
stoi["."] = 1
itos = Dict(reverse.(collect(stoi)))

block_size = 3
X, Y = [],Int[]
for w in words[1:5]
    @show w
    context = ones(Int, block_size)
    for c in w * "."
        c = string(c)
        ix = stoi[c]
        push!(X, context)
        push!(Y, ix)
        println(join(getd(itos, context)), "-->", itos[ix])
        context = cat(context[2:end], [ix];dims=1)
        # break
    end
    # break
end

X = stack(X, dims=1)
Y

C = randn(27, 2)
@test C[5, :] ≈ vec(onehot([5], 27) * C)
foo =  C[X]
C
emb = reshape(C[vec(X), :], (size(X, 1), 3, 2))
emb[1, :, :]
embb = permutedims(emb, (1, 3, 2))

W1 = randn(6, 100) # 6 because 3 sets of dim 2 embedding
b1 = randn(100)
@test_throws Any emb * W1 

# first emb
show(stdout, "text/plain", emb[1:10, 1, :])

input = reshape(emb, (size(X, 1), 6)) # WRONG! look at input[1, :]
#=
since the first X,Y pair  = "..." --> "e" for the start of emma
we expect the first row of the input to be 3 copies of the embedding for "."
    but thats not what we get
emb[1, :, :] -> 
emb[1, 1, :]


vec(emb[1, :, :]')

=#
@view emb[32, -1]

# this seems to work! 
embr = reshape(embb, (32, :))

(embr * W1) .+ b1