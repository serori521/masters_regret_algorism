# src/io_csv.jl
module IOCSV

using DelimitedFiles
using StringEncodings

"Shift_JIS の CSV を Float64 行列として読む（skipstart対応）"
function read_csv_f64_sjis(path::AbstractString; delim=',', skipstart::Int=0)
    io = open(path, enc"SHIFT_JIS", "r")
    data = readdlm(io, delim, Float64; skipstart=skipstart)
    close(io)
    return data
end

"通常のCSVを Float64 行列として読む"
read_csv_f64(path::AbstractString; delim=',') = readdlm(path, delim, Float64)

end # module
