function ix_maps(chars)
    stoi_ps = chars .=> eachindex(chars)
    stoi = Dict(stoi_ps)
    itos = Dict(reverse.(collect(stoi)))
    stoi, itos
end

t = read("C:/Users/anand/src/transcript_grabber/dataset.txt", String)

ps = zip(t, t[2:end]);
first(ps)
tal = tally(ps)

alpha = sort(unique(t))
nc = length(alpha)
keym = collect(Base.product(alpha, alpha))
d = Dict(vec(keym) .=> 0)
for x in ps
    d[x] += 1
end
function norm_rows(A; dims=2)
    A ./ sum(A; dims)
end

stoi, itos = ix_maps(alpha)

m = getd(d, keym)
heatmap(m)
reg = 0 
r = m .+ reg
char = '\n'
ix = stoi[char]
gen = []
for x in 1:1000
    ix =  sample(1:nc, Weights(r[ix, :]))
    push!(gen, itos[ix])
end
print(join(gen))