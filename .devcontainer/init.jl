using Pkg

# Get the activation path and Python version from command line arguments
PROJECT_PATH = ARGS[1]
PYTHON_VERSION = ARGS[2]

# Setup Python
Pkg.add("Conda")
using Conda
Conda.add("python=$PYTHON_VERSION")
Conda.add("jupyter")
Conda.add("nbconvert")

# Activate the project at the specified path
Pkg.activate(PROJECT_PATH)
Pkg.instantiate()
Pkg.precompile()