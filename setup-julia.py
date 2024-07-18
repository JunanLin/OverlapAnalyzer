# import os
# import juliacall
# from juliacall import Main as jl
import juliapkg
# juliacall.init(julia="/Users/junan/Documents/Research/Overlap/.venv/julia_env/julia-1.9.3/bin/julia")
# print(f"JULIA_HOME: {os.environ.get('JULIA_HOME')}")
# print(jl.seval('VERSION'))
# print(f"Julia is located at: {jl.seval('Sys.BINDIR')}")
# print("Julia executable: ", juliapkg.executable())
juliapkg.resolve()

# jl.seval('Pkg.activate("/Users/junan/Documents/Research/Overlap/.venv/julia_env/julia-1.9.3")')
# jl.seval('Pkg.add(Pkg.PackageSpec(;name="QuantumMAMBO", version="1.1.4"))')
# print(jl.seval('VERSION')) # Unchanged
# print(f"Julia is located at: {jl.seval('Sys.BINDIR')}")
# jl.seval('Pkg.add("QuantumMAMBO")')