
## Comments & Documentation

### Doc Comments
- Doc comments should begin with a short sentence that omits its subject, followed by a blank comment line.
  - For functions, that sentence should begin with a verb ("Returns the first element that...").
  - For other code, that sentence can also omit the verb ("The state of the waveform...").
- After a blank comment line, doc comments may continue with a more detailed explanation if necessary.
- When a short example is possible, an Example section should include that example.
- Doc comments should *not* describe previous versions of the code or changes that have occurred since previous versions of the code.
- Doc comments should *not* describe implementation details of the function; these comments should go inside the function.

### TODO Comments
- Comments that begin with TODO describe cases which are not handled or optimizations that should be considered at a later time.
- TODO comments should *not* be removed unless the code has been changed to address those cases or optimizations.

## Build & Test Verification
- After any Rust code changes, run `cargo build`, `cargo test`, `cargo fmt`, and `cargo clippy` before declaring work complete.
- For Jekyll/Ruby changes, run `bundle exec jekyll build` to verify clean build.

## Primary Stack
- Main language: Rust (use idiomatic patterns, prefer `Result` over panics, run `cargo fmt` and `cargo clippy` after each change).
- Secondary: Ruby/Jekyll for docs/site work.