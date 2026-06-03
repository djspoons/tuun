

## Build & Test Verification
- After any Rust code changes, run `cargo build` and `cargo test` before declaring work complete.
- For Jekyll/Ruby changes, run `bundle exec jekyll build` to verify clean build.

## Primary Stack
- Main language: Rust (use idiomatic patterns, prefer `Result` over panics, run `cargo fmt` and `cargo clippy` after each change).
- Secondary: Ruby/Jekyll for docs/site work.