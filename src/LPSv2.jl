module LPSv2

include("types.jl")
include("paths.jl")
include("io_csv.jl")
include("load_instance.jl")
include("init_state.jl")

make_instance(args...; kwargs...) = InitState.make_instance(read_utility_value, args...; kwargs...)
init_state(args...; kwargs...) = InitState.init_state(args...; kwargs...)

project_paths() = Paths.project_paths()

# re-export（呼びやすく）
read_utility_value(args...; kwargs...) = LoadInstance.read_utility_value(args...; kwargs...)
read_method_weights(args...; kwargs...) = LoadInstance.read_method_weights(args...; kwargs...)
read_true_weights(args...; kwargs...) = LoadInstance.read_true_weights(args...; kwargs...)

end
