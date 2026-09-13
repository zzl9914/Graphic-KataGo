from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

here = os.path.dirname(os.path.abspath(__file__))

ext = Pybind11Extension(
    "gkt_native",
    [
        "src/graph.cpp",
        "src/engine.cpp",
        "src/mcts.cpp",
        "src/selfplay.cpp",
        "src/torch_net.cpp",
        "bindings/gkt_native.cpp",
    ],
    include_dirs=[os.path.join(here, "include")],
    cxx_std=17,
)

setup(
    name="gkt_native",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
