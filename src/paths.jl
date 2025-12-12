# src/paths.jl
module Paths

"""
プロジェクト直下を基準に各フォルダの絶対パスを返す。
src/ から見て `..` がプロジェクトルート想定。
"""
function project_paths()
    root = normpath(joinpath(@__DIR__, ".."))
    return (
        root=root,
        src=joinpath(root, "src"),
        scripts=joinpath(root, "scripts"),
        data=joinpath(root, "data"),
        results=joinpath(root, "results"),
        tmp=joinpath(root, "tmp"),
        design=joinpath(root, "design"),
        archive=joinpath(root, "archive_old_lps"),
    )
end

end # module
