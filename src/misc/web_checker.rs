use std::collections::HashMap;
use std::fs;

use clap::Parser as ClapParser;

use tuun::{builtins, ids, modules, parser, slider};

#[derive(ClapParser, Debug)]
#[command(version, about = "Check tuun-synth expressions in .md and .html files")]
struct Args {
    #[arg(required = true)]
    input_files: Vec<String>,
}

type Bindings = Vec<parser::SourceBinding<ids::MarkId>>;

/// Builds the always-in-scope prelude: `sample_rate`, `tempo`, plus the
/// built-in definitions. Mirrors the wasm runtime's prelude so the
/// checker evaluates expressions the same way the browser would.
fn load_prelude() -> Bindings {
    fn def(id: &str, expr: parser::SourceExpr<ids::MarkId>) -> parser::SourceBinding<ids::MarkId> {
        parser::Binding::Definition(parser::Pattern::Identifier(id.to_string()), expr).into()
    }
    let mut bindings: Bindings = Vec::new();
    bindings.push(def("sample_rate", parser::SourceExpr::float(44100.0)));
    bindings.push(def("tempo", parser::SourceExpr::float(120.0)));
    builtins::add_bindings(&mut bindings);
    bindings
}

/// Parses every embedded module ahead of time and indexes them by dotted path.
/// The result feeds the `resolve` callback used by [`parser::evaluate`] when a
/// `Binding::Open` is encountered.
///
/// Each module gets an implicit `open __prelude` prepended so its bindings can
/// reference prelude names (`sample_rate`, `tempo`, built-ins) regardless of
/// caller ordering. Mirrors `evaluator::Evaluator::resolve` in the native
/// runtime and `Wasm::new` in the wasm bindings.
fn load_modules() -> HashMap<String, Bindings> {
    let mut out = HashMap::new();
    for (name, content) in modules::EMBEDDED_MODULES {
        match parser::parse_module::<ids::MarkId>(content) {
            Ok((mut bindings, errors)) => {
                if !errors.is_empty() {
                    eprintln!("Warning: failed to parse module '{}': {:?}", name, errors);
                }
                bindings.insert(
                    0,
                    parser::Binding::Open(vec!["__prelude".to_string()]).into(),
                );
                out.insert((*name).to_string(), bindings);
            }
            Err(e) => eprintln!("Warning: failed to parse module '{}': {:?}", name, e),
        }
    }
    out
}

/// Find the closing `>` of an HTML opening tag, skipping over quoted attribute values
/// that may contain `>` characters (e.g. `sliders='["freq:0.5:fn(x) => 100"]'`).
fn find_tag_close(html: &str) -> Option<usize> {
    let mut i = 0;
    let bytes = html.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'"' | b'\'' => {
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() && bytes[i] != quote {
                    i += 1;
                }
                // skip closing quote
                i += 1;
            }
            b'>' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Extract an attribute value from an HTML tag string.
/// Tries both double and single quotes.
fn extract_attr<'a>(html: &'a str, attr_name: &str) -> Option<&'a str> {
    for quote in ['"', '\''] {
        let pattern = format!("{}={}", attr_name, quote);
        if let Some(start) = html.find(&pattern) {
            let value_start = start + pattern.len();
            if let Some(end) = html[value_start..].find(quote) {
                return Some(&html[value_start..value_start + end]);
            }
        }
    }
    None
}

/// Strip tuun comments (// to end of line) from an expression.
fn strip_comments(expression: &str) -> String {
    expression
        .lines()
        .map(|line| {
            if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
        })
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Find all <tuun-synth> blocks in raw text and return (line_number, full_block_text) pairs.
fn find_tuun_synth_blocks(input: &str) -> Vec<(usize, &str)> {
    let mut blocks = Vec::new();
    let mut search_from = 0;

    while let Some(start) = input[search_from..].find("<tuun-synth") {
        let abs_start = search_from + start;
        let line = input[..abs_start].matches('\n').count() + 1;

        // Check for self-closing tag (ends with />)
        if let Some(tag_end) = input[abs_start..].find("/>") {
            let close_pos = abs_start + tag_end + 2;
            // But also check if there's a </tuun-synth> that comes before or after
            if let Some(close_tag) = input[abs_start..].find("</tuun-synth>") {
                let close_tag_pos = abs_start + close_tag + "</tuun-synth>".len();
                if close_tag_pos < close_pos {
                    // </tuun-synth> comes before /> — use content-based form
                    blocks.push((line, &input[abs_start..close_tag_pos]));
                    search_from = close_tag_pos;
                } else if tag_end < close_tag {
                    // /> comes first — self-closing
                    blocks.push((line, &input[abs_start..close_pos]));
                    search_from = close_pos;
                } else {
                    // </tuun-synth> comes first
                    blocks.push((line, &input[abs_start..close_tag_pos]));
                    search_from = close_tag_pos;
                }
            } else {
                // Only self-closing form
                blocks.push((line, &input[abs_start..close_pos]));
                search_from = close_pos;
            }
        } else if let Some(close_tag) = input[abs_start..].find("</tuun-synth>") {
            let close_tag_pos = abs_start + close_tag + "</tuun-synth>".len();
            blocks.push((line, &input[abs_start..close_tag_pos]));
            search_from = close_tag_pos;
        } else {
            // Malformed — skip past this occurrence
            search_from = abs_start + "<tuun-synth".len();
        }
    }

    blocks
}

/// Outcome of validating one `<tuun-synth>` block. `Skip` paths are
/// expected (no expression, malformed tag) — they print a `[skip]`
/// notice but don't count as failures. `Fail` paths print `[FAIL]` and
/// increment the failure tally.
enum CheckResult {
    Ok,
    Skip(String),
    Fail(String),
}

fn check_block(
    block: &str,
    prelude: &Bindings,
    modules: &HashMap<String, Bindings>,
) -> CheckResult {
    let description = extract_attr(block, "description").unwrap_or("");

    // Extract the expression — either from the `expression` attribute or
    // from the body between `<tuun-synth …>` and `</tuun-synth>`.
    let expression = match extract_expression(block) {
        Ok(s) => s,
        Err(message) => return CheckResult::Skip(message),
    };
    let expression = strip_comments(&expression);

    // Label that goes into log output. Prefer the description, otherwise
    // a flattened snippet of the expression.
    let label = if !description.is_empty() {
        description.to_string()
    } else {
        let flat: String = expression.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.len() > 60 {
            format!("{}...", &flat[..57])
        } else {
            flat
        }
    };

    let expr = match parser::parse_program::<ids::MarkId>(&expression) {
        Ok(e) => e,
        Err(errors) => {
            return CheckResult::Fail(format!("[FAIL] \"{}\" parse errors: {:?}", label, errors));
        }
    };

    // Sliders: `sliders='["volume:0.5:0:1", ...]'`.
    let slider_configs = match extract_attr(block, "sliders") {
        Some(s) => match parser::parse_sliders(&format!("sliders={}", s)) {
            Ok(cs) => cs,
            Err(e) => {
                return CheckResult::Fail(format!(
                    "[FAIL] \"{}\" slider parsing error: {:?}",
                    label, e
                ));
            }
        },
        None => vec![],
    };

    // Opens: `open='["std", "foo.bar"]'`. Same shape the wasm runtime
    // honors — each entry becomes a `Binding::Open(path)` resolved
    // against the embedded modules table.
    let open_attr = extract_attr(block, "open").unwrap_or("[]");
    let opens = match modules::parse_open_json(open_attr) {
        Ok(o) => o,
        Err(e) => {
            return CheckResult::Fail(format!("[FAIL] \"{}\" open parsing error: {}", label, e));
        }
    };

    // Build the bindings the same way the wasm runtime does:
    // implicit `open __prelude` → opens → sliders → expression.
    let mut bindings: Bindings = Vec::new();
    bindings.push(parser::Binding::Open(vec!["__prelude".to_string()]).into());
    for path in opens {
        bindings.push(parser::Binding::Open(path).into());
    }
    slider::append_slider_bindings(
        &slider_configs,
        &vec![0.0; slider_configs.len()],
        ids::MarkId::Slider,
        &mut bindings,
    );

    let resolve =
        |path: &[String]| -> Result<&[parser::SourceBinding<ids::MarkId>], parser::Error> {
            if path.len() == 1 && path[0] == "__prelude" {
                return Ok(prelude.as_slice());
            }
            let key = path.join(".");
            modules
                .get(&key)
                .map(|v| v.as_slice())
                .ok_or_else(|| parser::Error::new(format!("Module not found: {}", key)))
        };

    match parser::evaluate(resolve, &bindings, expr) {
        Ok(_) => CheckResult::Ok,
        Err(e) => CheckResult::Fail(format!("[FAIL] \"{}\" evaluate error: {:?}", label, e)),
    }
}

/// Pulls the expression out of a `<tuun-synth>` block. Tries the
/// `expression` attribute first, then falls back to the body (optionally
/// wrapped in `<script type="text/tuun">…</script>`).
fn extract_expression(block: &str) -> Result<String, String> {
    if let Some(expr) = extract_attr(block, "expression") {
        return Ok(expr.to_string());
    }
    let open_end = find_tag_close(block).ok_or_else(|| "malformed tag".to_string())?;
    let close_start = block
        .find("</tuun-synth>")
        .ok_or_else(|| "no expression".to_string())?;
    let content = block[open_end + 1..close_start].trim();
    // Strip an optional `<script type="text/tuun">…</script>` wrapper.
    let content = match content.find('>') {
        Some(script_end) if content[..script_end].contains("<script") => {
            let inner = &content[script_end + 1..];
            inner.strip_suffix("</script>").unwrap_or(inner).trim()
        }
        _ => content,
    };
    if content.is_empty() {
        return Err("no expression".to_string());
    }
    Ok(content.to_string())
}

fn check_file(
    file: &str,
    prelude: &Bindings,
    modules: &HashMap<String, Bindings>,
) -> (usize, usize) {
    let input = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  {} [error] {}", file, e);
            return (0, 1);
        }
    };
    let blocks = find_tuun_synth_blocks(&input);

    let mut found = 0;
    let mut failed = 0;

    for (line, block) in &blocks {
        found += 1;
        let label = extract_attr(block, "description").unwrap_or("");
        match check_block(block, prelude, modules) {
            CheckResult::Ok => println!("  {}:{} [ok] \"{}\"", file, line, label),
            CheckResult::Skip(msg) => {
                println!("  {}:{} [skip] \"{}\" ({})", file, line, label, msg)
            }
            CheckResult::Fail(msg) => {
                eprintln!("  {}:{} {}", file, line, msg);
                failed += 1;
            }
        }
    }

    if found > 0 {
        println!(
            "  {}: {} blocks, {} passed, {} failed",
            file,
            found,
            found - failed,
            failed
        );
    }

    (found, failed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let prelude = load_prelude();
    let modules = load_modules();

    let mut total_found = 0;
    let mut total_failed = 0;

    for file in &args.input_files {
        let (found, failed) = check_file(file, &prelude, &modules);
        total_found += found;
        total_failed += failed;
    }

    if args.input_files.len() > 1 {
        println!(
            "\ntotal: {} blocks, {} passed, {} failed",
            total_found,
            total_found - total_failed,
            total_failed
        );
    }

    if total_failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}
