# Coding standards

Rules for Rust in this repository. Each rule has an ID so reviews can cite it
("violates ERR-3"). The public-API bar is the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) (cited as `C-*`).
Rules marked **(lint)** are enforced by `[workspace.lints]` in `Cargo.toml`; the
rest are checked in review.

Each section opens with when it applies. When a change matches that trigger,
re-read the section before you write the code.

---

## 1. Invalid states (`IS`)

**Applies when** a domain type, enum, `bool`/`Option` flag, newtype, typestate,
or parse boundary is added or changed.

- **IS-1 Parse, don't validate.** Untrusted input (config, model output, user
  text, JSON) is converted once, at the boundary, into a type that can only hold
  valid values. Interior code accepts the validated type, never the raw `String`
  or `serde_json::Value`.
- **IS-2 Newtypes for domain identifiers and constrained values.** `ConversationId`,
  `ToolName`, `Temperature`, and the like are newtypes, not bare `String`/`u64`/`f32`.
  A constrained newtype's field is private, and its constructor (`new`/`TryFrom`/
  `FromStr`) returns `Result` when construction can fail.
- **IS-3 Enums over flag combinations.** Two or more `bool`/`Option` fields whose
  valid combinations are restricted become one `enum` with data-carrying variants.
  A struct with `is_streaming: bool, error: Option<E>, response: Option<R>` is a
  review failure.
- **IS-4 No sentinel values.** Do not use `""`, `0`, `-1`, or `u64::MAX` to mean
  "absent" or "unset". Use `Option` or a variant.
- **IS-5 Non-empty is a type.** If a collection must never be empty (for example
  the messages sent to a model), the type guarantees it with a `(first, rest)` pair
  or a newtype whose constructor rejects empty input.
- **IS-6 `bool` parameters.** Public functions do not take `bool` arguments that
  select behavior; use a two-variant enum so call sites read clearly (C-CUSTOM-TYPE).
- **IS-7 Exhaustive matches on our own enums.** Do not use a `_ =>` arm on enums
  defined in this workspace, so that adding a variant breaks every site that must
  handle it. Mark public enums that may grow `#[non_exhaustive]` only at crate
  boundaries we intend to publish.

```rust
// Bad: four fields, many impossible combinations.
struct ToolCall { started: bool, done: bool, output: Option<String>, error: Option<String> }

// Good: one field, every value is meaningful.
enum ToolCall {
    Pending { args: ToolArgs },
    Running { args: ToolArgs, started_at: Instant },
    Succeeded { output: ToolOutput },
    Failed { error: ToolError },
}
```

## 2. State machines (`SM`)

**Applies when** a workflow, lifecycle, pipeline, state machine, transition, or
state-dependent API is added or changed: agent turns, tool-call execution, streaming
responses, connection lifecycles, and UI modes.

- **SM-1 One enum per lifecycle.** The states are variants of one `enum`; each
  variant carries exactly the data that is valid in that state (see IS-3).
- **SM-2 Transitions are functions.** Changing state goes through methods that
  consume or match on the current state and return the next one, for example
  `fn on_event(self, event: Event) -> Result<State, TransitionError>`. Code outside
  the state machine's module does not assign variants directly.
- **SM-3 Illegal transitions are errors, not panics.** An event that is invalid in
  the current state returns a typed error (or is a no-op that the code documents).
  It never calls `unreachable!()`.
- **SM-4 Use typestate when the compiler can enforce order.** When a value
  passes through a fixed sequence of steps (builder, then connected, then ready),
  prefer a separate type per step (`Session<Connecting>` → `Session<Ready>`) so that
  calling a method in the wrong state does not compile. Use the enum form (SM-1)
  when the state is only known at runtime, such as when it is held in a GPUI view.
- **SM-5 Pure core, effects at the edge.** Transition functions do not perform IO.
  They return the next state plus a description of the effects to run (for example
  `Vec<Effect>`), and a driver executes those effects through ports (see PA). This
  keeps the agent loop testable without a network.
- **SM-6 Tests cover the transition table.** Every state machine has tests for each
  legal transition and at least one illegal transition per state.

## 3. Ownership (`OWN`)

**Applies when** a signature chooses between borrowing and owning, or code clones
or uses `Cow`.

- **OWN-1 Borrow in arguments by default.** Take `&str`, `&[T]`, `&T`, or
  `impl AsRef<_>` when the function only reads. Take ownership (`String`, `Vec<T>`,
  `T`) only when the function stores or consumes the value.
- **OWN-2 Ask for owned values when you must store them.** If a function stores a
  value, take `T` (or `impl Into<T>`) rather than `&T` followed by an internal
  `.clone()`, so callers that already own the value can move it in (C-CALLER-CONTROL).
- **OWN-3 Justify `.clone()`.** Clones in hot paths, such as per streamed token or
  per frame, need a reason. Cheap handle clones (`Arc`, GPUI `Entity`, channel
  senders) are fine and should be written as `Arc::clone(&x)` **(lint:
  `clone_on_ref_ptr`)**.
- **OWN-4 `Cow` only for real conditional ownership.** Use `Cow<'_, str>` when a
  function usually returns borrowed data and only sometimes allocates. Do not use it
  as a general way to avoid thinking about lifetimes.
- **OWN-5 Return owned data from constructors and parsers.** Public types do not
  hold borrowed lifetimes unless they are clearly short-lived views, such as
  iterators or parsers over a buffer.

## 4. Errors (`ERR`)

**Applies when** a function is fallible, or code adds error types, context,
`unwrap`, or `expect`.

- **ERR-1 Typed errors at library boundaries.** Every public fallible function in a
  library crate (`harness-core` and future libs) returns `Result<T, E>` where `E` is
  a concrete error type deriving `thiserror::Error`. Do not return `anyhow::Error`,
  `Box<dyn Error>`, or `String` from library APIs.
- **ERR-2 One error enum per module or concern**, not one for the whole crate. Variants
  describe what failed (`ModelRequest`, `ToolNotFound { name }`), not the type of
  the underlying error.
- **ERR-3 Keep the source chain.** Wrap underlying errors with `#[source]` or
  `#[from]`. Do not stringify them with `.to_string()` or `format!("{e}")`.
  `Display` messages are lowercase, have no trailing period, and do not repeat the
  source's message (C-GOOD-ERR).
- **ERR-4 `anyhow` only in the binary.** `harness-app` may use `anyhow::Result`
  with `.context(...)` at the top level, where errors are shown to the user or
  logged.
- **ERR-5 No panics outside tests (lint).** `unwrap`, `expect`, `panic!`, `todo!`,
  `unimplemented!`, `dbg!` are denied in non-test code; `clippy.toml` allows them
  inside `#[cfg(test)]`. Index with `.get()` instead of `[]` **(lint:
  `indexing_slicing`)**. A genuine invariant violation is either a typed error or,
  if truly impossible, restructured so the type system proves it (see IS).
- **ERR-6 Document failures.** Public fallible functions have an `# Errors` doc
  section listing when each variant is returned **(lint: `missing_errors_doc`)**.
- **ERR-7 Do not swallow errors.** `let _ = fallible();` and `.ok()` that discard an
  error need a comment saying why ignoring it is correct.

```rust
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model request failed")]
    Request(#[source] reqwest::Error),
    #[error("model returned malformed response")]
    Decode(#[from] serde_json::Error),
    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Duration },
}
```

## 5. Traits and generics (`TG`)

**Applies when** code defines a trait, a generic function, a `dyn` object, or a
conversion.

- **TG-1 Traits exist for a reason.** Define a trait only for a real substitution
  point: a port (see PA), multiple concrete implementations, or an extension point.
  Do not add one-impl traits "for testability" unless the trait is a port.
- **TG-2 Generics by default, `dyn` when the value is held at runtime.** Use
  `impl Trait`/generics in function arguments. Use `Box<dyn Trait>`/`Arc<dyn Trait>`
  when heterogeneous values are stored, such as a tool registry, or to keep
  generics out of GPUI view types.
- **TG-3 Async traits.** Port traits that perform async IO use native `async fn`
  in traits. When they must be object-safe or `Send`, return
  `impl Future<Output = _> + Send` or use explicit boxed futures. Choose one style
  per crate and use it everywhere.
- **TG-4 Standard conversions.** Implement `From`/`TryFrom`/`FromStr`/`Display`
  instead of ad-hoc `to_x`/`from_x` methods when the standard trait fits (C-CONV-TRAITS).
  `From` is only for infallible, lossless conversions.
- **TG-5 Common derives on public types.** Public types implement `Debug`
  **(lint: `missing_debug_implementations`)**. They also implement `Clone`,
  `PartialEq`, `Eq`, `Hash`, and `Default` wherever those traits are meaningful
  (C-COMMON-TRAITS). Do not derive `Default` when the default value would be invalid.
- **TG-6 Keep bounds minimal.** Put trait bounds on the `impl` blocks or functions
  that need them, not on struct definitions.

## 6. Builders and constructors (`BLD`)

**Applies when** a struct gains many options, or a public type needs a
constructor.

- **BLD-1 `new` takes only required data.** Use `new(required...)` for the
  required fields and keep everything else at defaults. More than about three
  required arguments, or any optional configuration, calls for a builder.
- **BLD-2 Builders validate in `build()`.** `build()` returns `Result<T, BuildError>`
  when the configuration can be invalid. The built type is always valid (see IS-1).
- **BLD-3 Builder methods take and return `self` by value** (`fn model(mut self, m: Model) -> Self`)
  for one-shot configuration, so calls can be chained.
- **BLD-4 Do not expose public fields on types with invariants.** Use private
  fields plus accessors. Plain data types without invariants (DTOs, events) may
  have public fields.

## 7. Iterators (`IT`)

**Applies when** code loops over a collection or collects results.

- **IT-1 Prefer iterator adapters** (`map`, `filter`, `find`, `any`, `fold`) over
  manual index loops. Use a `for` loop when the body has side effects or early
  returns that adapters would obscure.
- **IT-2 Collect fallible results with `collect::<Result<Vec<_>, _>>()`** rather than
  pushing inside a loop and checking errors by hand.
- **IT-3 Do not collect only to iterate again.** Return `impl Iterator<Item = _>`
  from internal helpers when callers only iterate.
- **IT-4 Do not index inside loops (lint: `indexing_slicing`).** Iterate the
  values directly or use `.iter().enumerate()`, `.zip()`, or `.windows()`.

## 8. Concurrency and async (`CON`)

**Applies when** code uses threads, channels, shared state, async, or a network
client.

- **CON-1 Never block the UI thread.** GPUI views do not perform IO or other
  long-running work inline. They spawn work on GPUI's background executor
  (`cx.background_spawn`) or hand it to core through a channel, and update
  entities when results arrive.
- **CON-2 Message passing over shared mutable state.** Prefer channels between
  the agent loop and the UI. If shared state is unavoidable, keep the lock scope
  minimal and never hold a lock across `.await`.
- **CON-3 Cancellation is explicit.** Long-running operations (model streams, tool
  runs) accept a cancellation signal or are tied to a task handle that is dropped
  to cancel. A stop action in the UI must actually stop the work.
- **CON-4 Timeouts on all network calls.** Every model and HTTP call has a
  timeout. Retries use bounded exponential backoff and are part of the port
  adapter, not scattered through callers.
- **CON-5 Core is runtime-agnostic where practical.** `harness-core` exposes async
  functions and does not start its own runtime. The app decides how futures are
  executed. If an adapter needs a specific runtime (for example tokio for
  `reqwest`), document that requirement on the adapter.
- **CON-6 `Send + 'static` bounds are deliberate.** Add them where tasks cross
  threads, and say why in the docs of public APIs that require them.

## 9. Unsafe (`UNS`)

**Applies when** code uses `unsafe`, FFI, raw pointers, or `transmute`.

- **UNS-1 `unsafe` is forbidden (lint: `unsafe_code = "forbid"`).** This cannot be
  overridden locally. If a dependency's API truly requires it, raise it in the PR
  description, isolate it in a dedicated crate with its own lint config, and give
  every block a `// SAFETY:` comment **(lint: `undocumented_unsafe_blocks`)**.

## 10. Modules, visibility, docs (`MOD`)

**Applies when** code adds a module or file, sets visibility, or writes doc
comments.

- **MOD-1 Smallest visibility that works.** Default to private, then `pub(crate)`,
  and use `pub` only for intended API. Write `pub(crate)` rather than `pub` for
  items in private modules **(lint: `unreachable_pub`)**.
- **MOD-2 Organize modules by domain** (`conversation`, `tools`, `model`), not by
  kind (`types`, `utils`, `helpers`). Use `foo.rs` + `foo/` rather than `foo/mod.rs`.
- **MOD-3 Every public item is documented (lint: `missing_docs`).** The first line
  is a one-sentence summary. Fallible functions have `# Errors` and functions that
  can panic have `# Panics` (which should be rare; see ERR-5). Add `# Examples`
  for core entry points, because examples are compiled and run as doctests.
- **MOD-4 Doc links must resolve (lint: rustdoc `broken_intra_doc_links`, CI runs
  `cargo doc` with `-D warnings`).**
- **MOD-5 Re-export deliberately.** A crate's `lib.rs` re-exports its public API
  surface. Do not use glob re-exports (`pub use foo::*`).

## 11. Ports, adapters, and testing (`PA`)

**Applies when** tests cover code that calls a database, file system, HTTP service,
or mock.

- **PA-1 IO behind ports.** Code that reaches outside the process (model APIs,
  file system, clock, process spawn, environment) does so through a trait defined
  in core, for example `trait ModelClient`, `trait ToolExecutor`,
  `trait ConversationStore`, or `trait Clock`. Domain logic depends on the trait,
  never on `reqwest`/`std::fs` directly.
- **PA-2 Adapters are thin.** An adapter translates between the port's domain
  types and the external API (serialization, HTTP, error mapping) and contains no
  business logic.
- **PA-3 Fakes over mocks.** Tests use hand-written in-memory fakes, such as a
  `FakeModelClient` that replays scripted responses or a `MemoryStore` backed by a
  `Vec`. Fakes are reusable, live in a `#[cfg(test)]` module or a `testing` module
  behind a feature, and assert on resulting state instead of call sequences. Use
  mocking libraries only when a fake would be unreasonably complex.
- **PA-4 No network in `cargo test`.** Unit and integration tests run offline and
  are deterministic. Tests against real services are `#[ignore]`d and say in
  their docs which environment variables they need.
- **PA-5 Inject time and randomness.** Code that depends on the current time or on
  random values takes them through a port, so tests are deterministic.
- **PA-6 Contract tests for adapters.** When a port has a fake and a real
  adapter, a shared test suite (a generic function over `impl Port`) runs against
  the fake in CI and against the real adapter in the ignored tests.
- **PA-7 Test names describe behavior**: `rejects_empty_message`, not `test1`.
  Each test has one reason to fail.

---

## Lint overrides

Workspace lints live in the root `Cargo.toml`. Crates opt in with
`[lints] workspace = true` and do not redefine them.

- Prefer `#[expect(lint, reason = "...")]` over `#[allow]`. `expect` warns when the
  lint stops firing, so stale overrides get removed **(lint: `allow_attributes`)**.
- Every override has a `reason` that names the guideline or rule it departs from
  and why **(lint: `allow_attributes_without_reason`)**. For example:
  `#[expect(clippy::cast_possible_truncation, reason = "token counts fit in u32; see model limits")]`.
- Scope overrides to the smallest item. Crate-level overrides and changes to
  `[workspace.lints]` need a justification in the PR description.
- Workspace-level opt-outs (currently only `must_use_candidate`) are documented
  inline in `Cargo.toml`.

## Formatting

`rustfmt.toml` is authoritative; run `cargo fmt --all`. Do not hand-format code
against it or add `#[rustfmt::skip]` without a reason.
