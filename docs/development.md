# Development

Start with the CPU core when changing selection, storage, ownership or request
planning. GPU behavior requires a separate model/runtime environment and should
be validated only when the change touches that path.

## CPU-only setup

From the repository root, with Python 3.12 and a C++17 compiler available:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest \
  tests/test_core.py tests/test_nvme.py tests/test_storage.py -q
```

This builds the pybind11 host extension and installs pytest. It does not install
torch or vLLM. The command above matches the focused Python selection in the
[host-core workflow](../.github/workflows/host-core.yml).

Build and run the C++ tests with CMake 3.20+:

```sh
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core -j2
ctest --test-dir build/core --output-on-failure
```

Use a Debug build for these assertion-based checks. Build outputs, virtual
environments, compiled extensions and experiment artifacts are ignored by Git.
After changing native extension code, reinstall the editable package to rebuild
it before running Python tests.

## Source layout

| Path | Purpose |
| --- | --- |
| `src/vllm_kvmem/` | Python serving entry points, runtime adapters, request/session coordination and storage integration |
| `csrc/` | Host bindings, historical-memory policies and CUDA source used by runtime components |
| `third_party/qw3/` | Imported host core, tests and provenance records |
| `third_party/vllm/` | vLLM license/provenance material |
| `tests/` | Host regressions, interface checks and opt-in GPU/model checks |
| `examples/` | Explicit serving and experiment profiles |
| `scripts/` | Setup, qualification, measurement and artifact utilities |
| `benchmarks/deepswe/` | DeepSWE integration and its separate benchmark documentation |

The host core is built by `setup.py`; CUDA source files needed by the runtime
are included in the package. Keep source provenance and third-party license
notices when adapting imported code.

## Serving and compatibility checks

For GPU dependencies and the first request, follow
[installation](installation.md). Use a separate checkout/environment when
comparing dependency versions so a running measurement retains its recorded
runtime.

The launcher supports read-only inspection without importing torch or vLLM:

```sh
PYTHONPATH=src python3.12 -m vllm_kvmem serve --help
PYTHONPATH=src python3.12 -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/inspect-only --dry-run
```

A dry run still requires valid local model paths. It does not create the run
directory or exercise GPU kernels.

To check an upstream vLLM source tree without loading its GPU implementation:

```sh
PYTHONPATH=src python3.12 -m vllm_kvmem.engine_api \
  --source /absolute/path/to/vllm/vllm --systemone
```

Omit `--systemone` when checking the historical baseline without that optional
API. See [compatibility](vllm-compatibility.md) and the
[source-compatibility workflow](../.github/workflows/vllm-compat.yml) for the
difference between structural checks, CPU regressions and model execution.

Run the existing focused tests for the component you change. The full test tree
includes tests with optional dependencies and hardware requirements, so a
CPU-only install is not a promise that every test can run in that environment.
Add a regression test when a behavioral fix needs to protect an invariant;
documentation-only changes should be checked through their links, examples and
command syntax.

## Runtime invariants

- Keep logical history positions, selected dense positions, physical GPU pages
  and generation-qualified archive handles distinct.
- Publish or retire state only after its required transfer/ownership fences.
  Pending readers must not observe reused pages or a different session generation.
- Recurrent state and speculative state must describe the same accepted prefix
  as the restored attention KV. Rejected draft tokens must not become history.
- Private query/scoring work must not become public output or modify another
  request's state.
- Retain explicit runtime/interface checks for unsupported layouts. A source
  adaptation must not silently skip ownership, shape or generation validation.

The [engine integration](engine-integration.md),
[request sessions](prompt-sessions.md), and
[rolling-output design](rolling-long-output.md) provide additional context.

## Benchmark changes

Record the source revision, explicit profile, dependency versions, hardware,
input provenance and timing boundaries with a measurement. Keep full input
histories and output quotas fixed when comparing serving throughput. Check
streamed output counts against server usage/counters, and include actual GPU
memory measurements when making a capacity claim.

Compare native vLLM and KVMem with the same model, precision and intended
workload. Any necessary allocation-budget differences must be recorded together
with actual memory usage. History preparation, continuation generation, pure
decode intervals and end-task success are different measurements; label the one
being reported. Replaying captured output in a visualization does not constitute
additional completed inference work.

Do not replace a failed or interrupted measurement with a success record. Keep
its cause and provenance, and record a subsequent attempt independently.

## Contributing

Describe the behavior changed, why it is needed, and the focused validation
performed. Include the model/runtime configuration for GPU changes. Avoid adding
weights, private model paths, credentials or local measurement directories to a
patch. Keep raw benchmark artifacts outside the source tree's tracked files and
link a compact, self-contained result when needed.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the short contribution workflow and
[support](support.md) for issue-report details.

## Historical development record

The original development journal, its dated acceptance tables, failed attempts
and evidence corrections are preserved in
[the development archive](archive/development-before-public-guide.md). Those
entries are useful for tracing decisions, but their old "next run" instructions
are not current setup steps.
