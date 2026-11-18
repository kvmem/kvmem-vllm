# Project development requirements

Follow the project's [contribution guidelines](CONTRIBUTING.md) and
[development setup](docs/development.md).

## Parallel development

Keep the primary checkout on `main` for integration. Give each independent
task that runs in parallel its own Git worktree and short-lived branch. Start
independent tasks from an up-to-date `main` and name branches after the change.
Use one editing session per worktree. When switching between Codex and Claude
Code on the same task, hand off the existing worktree before the next session
starts editing.

- In the Codex desktop app, start a chat in **Worktree** mode from `main`.
  Use **Create branch here** to name the branch before committing.
  [Codex worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees)
- In Claude Code, run `claude --worktree bench-report` from the main checkout
  to start a session on its own branch. Use a different name for each task.
  [Claude Code worktrees](https://code.claude.com/docs/en/worktrees)

To include commits that exist only on local `main`, create the worktree with
Git. Run this command from the main checkout, then open the new directory in
Codex or start Claude Code from that directory.

```sh
git worktree add -b codex/decode-metrics ../kvmem-vllm-decode-metrics main
```

## Builds and benchmarks

Build native extensions in the worktree under test and verify that the test
environment imports that worktree's `vllm_kvmem` package and extensions. Share
read-only model weights and datasets. Use a separate output directory for
each run and separate ports for servers on the same host. Run performance
benchmarks one at a time on a shared GPU, or assign separate GPUs to
concurrent benchmarks.

## Integration and cleanup

1. Commit the changes on the task branch. Report the commit, validation
   results, and the location of any benchmark outputs.
2. Have one session review and merge task branches into `main` one at a time.
   Run the relevant regression checks on the merged `main`.
3. Move benchmark artifacts you want to keep to a directory outside the
   worktree before cleaning it up. Use the managing app's cleanup for
   worktrees it created, or `git worktree remove` for a manually created
   worktree. After confirming the branch is merged and releasing its
   checkout, delete its local name with `git branch -d`.
