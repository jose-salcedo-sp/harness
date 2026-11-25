# AGENTS.md

Instructions for coding agents (and humans) working in this repository.

## What this is

`harness` is a personal AI agent harness: a desktop chatbot window built with
[GPUI](https://www.gpui.rs/) and [gpui-component](https://github.com/longbridge/gpui-component),
backed by an agent loop written from scratch in Rust.

## Layout

| Path | Crate | Role |
| --- | --- | --- |
| `crates/harness-core` | `harness-core` (lib) | Domain types, agent loop, ports (traits) for models, tools, storage. **No UI dependencies.** |
| `crates/harness-app` | `harness-app` (bin `harness`) | GPUI window: views, input, rendering. Depends on `harness-core`, never the reverse. |

Dependency rules:

- `harness-core` must not depend on `gpui`, `gpui-component`, or any windowing crate.
  If core needs to notify the UI, it exposes a channel, stream, or callback trait.
- Real IO (HTTP model clients, file system, process spawning) lives behind a port
  trait defined in core. Concrete adapters live in core behind a module, or in a new
  crate when they pull heavy dependencies. Tests use in-memory fakes.
- New crates go under `crates/`, inherit `[workspace.package]` fields, and set
  `[lints] workspace = true`. Shared dependency versions go in `[workspace.dependencies]`.

## Done check (run all before you push)

CI (`.github/workflows/ci.yml`) is the source of truth. Locally:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo test --workspace --doc --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
```

`--all-targets` does not run doctests, which is why `--doc` is listed separately.
If you switch to `cargo nextest`, keep the `cargo test --doc` step. Drop `--locked`
only for the one command that updates `Cargo.lock` after you add a dependency,
then commit the lockfile.

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it automatically.

### Building the app on Linux

GPUI needs system libraries. On Debian/Ubuntu, install the same packages CI does
(see the "Install GPUI system dependencies" step in `.github/workflows/ci.yml`).

## Rules

Follow [`CODING_STANDARDS.md`](CODING_STANDARDS.md). The short version:

1. Make invalid states unrepresentable; parse at the boundary into validated types.
2. Model lifecycles (agent turns, tool calls, streaming) as explicit state machines.
3. Libraries return typed errors (`thiserror`); the binary may use `anyhow`.
4. No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, or `dbg!` outside
   tests. Clippy denies them.
5. IO goes through ports; tests use fakes, not the network.
6. Every public item has a doc comment; fallible public functions document `# Errors`.
7. No `unsafe` (forbidden workspace-wide).
8. Every `#[allow]`/`#[expect]` carries a `reason = "..."`; prefer `#[expect]`.

### Scope

- **Feature or fix:** apply the standards to code you add or whose contracts you
  change. Do not refactor unrelated code in the same PR.
- **Audit or cleanup task:** apply them to all named code; report each violation
  with the rule it breaks, or fix it if asked.

## Working conventions

- Keep PRs focused; one behavior change per PR where practical.
- Add or update tests with every behavior change. Bug fixes start with a failing test.
- Do not loosen a workspace lint to make code pass. Fix the code, or add a scoped
  `#[expect(lint, reason = "...")]` that names the guideline you are departing from.
- Do not commit secrets. API keys come from the environment at runtime.
