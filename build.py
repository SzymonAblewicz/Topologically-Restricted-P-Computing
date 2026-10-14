"""
Compile the four C++ cores in place, next to their sources, with pybind11 (MSVC on
Windows, g++ on Linux). PSSA and Divide and Conquer need them to run. Also compiles
stream_validate, the validator of the large fallback embeddings.

    python build.py
"""
import os
import tempfile

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

ROOT = os.path.dirname(os.path.abspath(__file__))
CORES = [
    "src/pssa/improved_pssa/cheap_pssa_core",                      # the improved PSSA
    "src/pssa/pssa_wbc/cheap_pssa_core",                           # PSSA wBC
    "src/divide_and_conquer/floorplan_core",                       # placing and routing the clusters
    "experiments/section2_pssa/translation_to_cpp/pssa_core",      # the first C++ loop, for the speed comparison
]
VALIDATOR = "experiments/section4_fallback/validation/stream_validate"

if __name__ == "__main__":
    os.chdir(ROOT)
    ext = [Pybind11Extension(c.replace("/", "."), [c + ".cpp"], cxx_std=17) for c in CORES]
    with tempfile.TemporaryDirectory() as tmp:
        setup(name="cores", ext_modules=ext, cmdclass={"build_ext": build_ext},
              script_args=["build_ext", "--inplace", "--build-temp", tmp,
                           "--build-lib", os.path.join(tmp, "lib")])
        cc = new_compiler()
        customize_compiler(cc)
        flags = ["/O2", "/std:c++17", "/EHsc"] if cc.compiler_type == "msvc" else ["-O2", "-std=c++17"]
        objs = cc.compile([VALIDATOR + ".cpp"], output_dir=tmp, extra_preargs=flags)
        cc.link_executable(objs, os.path.basename(VALIDATOR), output_dir=os.path.dirname(VALIDATOR),
                           target_lang="c++")
