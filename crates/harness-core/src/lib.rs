//! UI-free backend for the harness chatbot: domain types, the agent loop, and
//! the ports (traits) through which it reaches models, tools, and storage.
//!
//! This crate must not depend on GPUI or any other UI toolkit. Adapters that
//! perform real IO implement the ports defined here; tests use in-memory fakes.
//! See `CODING_STANDARDS.md` at the repository root for the rules this crate
//! follows.
