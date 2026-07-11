
## Comments & Documentation

### Doc Comments
- Doc comments should begin with a short sentence that omits its subject, followed by a blank comment line.
  - For functions, that sentence should begin with a verb ("Returns the first element that...").
  - For other code, that sentence can also omit the verb ("The state of the waveform...").
- After a blank comment line, doc comments may continue with a more detailed explanation if necessary.
- When a short example is possible, an Example section should include that example.
- Doc comments should *not* describe previous versions of the code or changes that have occurred since previous versions of the code.
- Doc comments should *not* describe implementation details of the function; these comments should go inside the function.
- Doc comments should describe only the item being documented — its behavior, inputs, outputs, and preconditions — *not* the behavior of its callers or of other parts of the codebase. Describing callers inverts the dependency: a later change to the caller would require an edit to this comment, and that edit will be missed.
  - If the reader needs to know *when* to call a function, state it as a precondition or condition the function itself can express ("Call once the text's source identity is known"), not as a survey of current call sites ("the parser does X, and then module loading calls this").
  - Facts about how another component behaves belong in that component's own docs; use a cross-reference link if needed rather than restating them.

### TODO Comments
- Comments that begin with TODO describe cases which are not handled or optimizations that should be considered at a later time.
- TODO comments should *not* be removed unless the code has been changed to address those cases or optimizations.

## Build & Test Verification
- After any Rust code changes, run `cargo build`, `cargo build --benches`, `cargo test`, `cargo fmt`, and `cargo clippy` before declaring work complete. (Benches are not compiled by `cargo build` or `cargo test`, so they break silently without the `--benches` check.)
- For Jekyll/Ruby changes, run `bundle exec jekyll build` to verify clean build.

## Primary Stack
- Main language: Rust (use idiomatic patterns, prefer `Result` over panics, run `cargo fmt` and `cargo clippy` after each change).
- Secondary: Ruby/Jekyll for docs/site work.

## Rust Style

### Imports
- Bring crate-internal dependencies into scope with `use crate::<module>` (or `use crate::<module>::{self, Item}`) at the top of the file, then write `module::item` or `Item` at use sites. Do *not* write inline `crate::...` paths in signatures or bodies — the `use` block should read as the file's complete dependency list.
- The same applies inside `#[cfg(test)]` modules: add `use crate::...` lines to the test module rather than writing inline paths.
- Rustdoc link targets (e.g. ``[`crate::parser::parse_module`]``) are exempt — they are documentation cross-references, not code dependencies.

## UI Conventions

### Program identifiers in user-visible text
- User-visible strings that name a program (log lines, status messages, `println!` output) must go through `programs::ProgramSet::display_name(program_index)` (e.g. `state.programs.display_name(i)`). Do NOT interpolate a raw program index or `index + 1`.
- The helper prefers the binding's name (e.g. `kick`) and falls back to a bank-relative address like `B:3` — the letter is the bank (A..H), the digit is the 1-based slot within the bank (matching the digit the user types to select it).
- Rationale: raw indices span 1..64 across 8 banks of 8, so they don't match the keystroke the user would use (e.g. `1` on bank 2 is index 9). Bank-relative addresses are what the user expects to see in response.
- Internal/debug logs about source parsing (e.g. "Ignoring program with out-of-range slot N") can still show the raw slot number — that's what the source file declares.