from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from pathlib import Path
from shutil import copyfile

# CUDA JIT sources travel with the wheel as well as the sdist.
gpu_sources = Path('src/vllm_kvmem/cuda_sources')
gpu_sources.mkdir(exist_ok=True)
for name in ('qw3_mean_k.cu', 'qw3_mean_k_kernel.cuh', 'qw3_mean_k_tiled.cuh', 'qw3_mean_k_provenance.json'):
    copyfile(Path('csrc') / name, gpu_sources / name)

setup(
    package_data={'vllm_kvmem': ['cuda_sources/*.cu', 'cuda_sources/*.cuh', 'cuda_sources/*.json',
                                'native*.cuh', 'native_attention_sources.json']},
    ext_modules=[Pybind11Extension(
        "vllm_kvmem._native",
        ["csrc/bindings.cpp", "csrc/historical_payload.cpp", "csrc/historical_manager.cpp",
         "csrc/historical_redundancy.cpp"] + [
            f"third_party/qw3/src/{name}.cpp" for name in
            ("kvmem_store", "kvmem_repository", "kvmem_repository_gc", "kvmem_gc")
        ],
        include_dirs=["third_party/qw3/include", "third_party/qw3/src"],
        depends=["csrc/private_record_pack.hpp", "csrc/graph_propagation.hpp", "csrc/restore_frame.hpp",
                 "csrc/query_key_fps_cpu.hpp", "csrc/historical_kv_policy.hpp",
                 "csrc/historical_policy_factory.hpp", "csrc/historical_manager.hpp",
                 "csrc/historical_redundancy.hpp"],
        cxx_std=17,
        extra_compile_args=["-pthread"],
        extra_link_args=["-pthread"],
    )],
    cmdclass={"build_ext": build_ext},
)
