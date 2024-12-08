using Flux, OneHotArrays, ProgressMeter, StatsBase, Plots, Random, BenchmarkTools, LinearAlgebra
using Graphs
using Flux.Losses
using Flux: glorot_uniform, kaiming_normal

# https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#Attention
