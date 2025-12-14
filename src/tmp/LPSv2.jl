# src/LPSv2.jl
module LPSv2

include("types.jl")        # => module CoreTypes
include("paths.jl")        # => module Paths
include("io_csv.jl")       # => module IOCSV
include("load_instance.jl")# => module LoadInstance
include("init_state.jl")   # => module InitState

# 外から使う入口（今のまま）
project_paths() = Paths.project_paths()

read_utility_value(args...; kwargs...) = LoadInstance.read_utility_value(args...; kwargs...)
read_method_weights(args...; kwargs...) = LoadInstance.read_method_weights(args...; kwargs...)
read_true_weights(args...; kwargs...) = LoadInstance.read_true_weights(args...; kwargs...)

make_instance(args...; kwargs...) = InitState.make_instance(read_utility_value, args...; kwargs...)
init_state(args...; kwargs...) = InitState.init_state(args...; kwargs...)

# 型も外から使えるように（必要なら）
export CoreTypes, project_paths, read_utility_value, read_method_weights, read_true_weights, make_instance, init_state

end # module
