# Contributing

Changes that improve correctness, usability, runtime compatibility or documented
performance are welcome. Start with [installation](docs/installation.md) for GPU
serving or [development](docs/development.md) for the CPU-only environment.

1. Describe the problem and the expected behavior in an issue or pull request.
2. Keep the change focused. Use an explicit local profile when a reproduction
   depends on a model, hardware or runtime configuration.
3. Run the relevant existing checks. Include a regression test for behavioral
   fixes, and distinguish CPU/interface checks from actual-model GPU validation.
4. Update the user-facing instructions when an entry point, configuration or
   supported behavior changes.
5. In the pull request, explain the result, validation and remaining limitations.

For performance changes, provide comparable input/output counts, timing
boundaries, memory usage and native-baseline settings. A throughput increase is
not evidence of unchanged task quality; report those checks separately.

Keep third-party provenance and license notices intact. Do not commit model
weights, credentials, machine-specific local profiles or the contents of
`artifacts/` and `runs/`. The project is distributed under
[Apache-2.0](LICENSE); third-party components retain their included notices.

For bug reports, include the details described in
[Support and compatibility](docs/support.md#reporting-an-issue).
