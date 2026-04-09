use std::fs;

use clap::Parser as ClapParser;

use tuun::{builtins, parser, renderer, slider};

#[derive(ClapParser, Debug)]
#[command(version, about = "Check tuun-synth expressions in .md and .html files")]
struct Args {
    #[arg(required = true)]
    input_files: Vec<String>,
}

fn load_context() -> Vec<(String, parser::Expr<renderer::MarkId>)> {
    // TODO this context business is now getting duplicated across main, here, and the web stuff
    let mut context = Vec::new();
    context.push(("sample_rate".to_string(), parser::Expr::Float(44100.0)));
    context.push(("tempo".to_string(), parser::Expr::Float(120.0)));
    builtins::add_prelude(&mut context);

    let context_content = include_str!("../../context.tuun");
    let context_content: String = context_content
        .lines()
        .map(|line| {
            if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
        })
        .collect::<Vec<&str>>()
        .join("\n");

    match parser::parse_context(&context_content) {
        Ok(parsed_defs) => {
            for (pattern, expr) in parsed_defs {
                match parser::evaluate(&context, expr) {
                    Ok(expr) => {
                        if let Err(e) = parser::extend_context(&mut context, &pattern, &expr) {
                            eprintln!("Warning: Failed to add context definition: {:?}", e);
                        }
                    }
                    Err(e) => eprintln!("Warning: Failed to evaluate context: {:?}", e),
                }
            }
        }
        Err(e) => eprintln!("Warning: Failed to parse context file: {:?}", e),
    }

    context
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

fn check_file(
    file: &str,
    context: &Vec<(String, parser::Expr<renderer::MarkId>)>,
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
        let description = extract_attr(block, "description").unwrap_or("");

        // Extract expression: from attribute or from content between tags
        let expression = if let Some(expr) = extract_attr(block, "expression") {
            expr.to_string()
        } else {
            // Find the closing > of the <tuun-synth ...> opening tag,
            // skipping > chars inside quoted attribute values
            let open_end = match find_tag_close(block) {
                Some(pos) => pos,
                None => {
                    println!(
                        "  {}:{} [skip] \"{}\" (malformed tag)",
                        file, line, description
                    );
                    continue;
                }
            };
            if let Some(close_start) = block.find("</tuun-synth>") {
                let content = &block[open_end + 1..close_start];
                let content = content.trim();
                // Strip <script type="text/tuun">...</script> wrapper if present
                let content = if let Some(script_end) = content.find('>') {
                    if content[..script_end].contains("<script") {
                        let inner = &content[script_end + 1..];
                        inner.strip_suffix("</script>").unwrap_or(inner).trim()
                    } else {
                        content
                    }
                } else {
                    content
                };
                if content.is_empty() {
                    println!(
                        "  {}:{} [skip] \"{}\" (no expression)",
                        file, line, description
                    );
                    continue;
                }
                content.to_string()
            } else {
                println!(
                    "  {}:{} [skip] \"{}\" (no expression)",
                    file, line, description
                );
                continue;
            }
        };

        let expression = strip_comments(&expression);

        // Use description if available, otherwise show the start of the expression
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

        // Parse and evaluate
        match parser::parse_program(&expression) {
            Ok(expr) => {
                let configs = if let Some(configs) = extract_attr(block, "sliders") {
                    match parser::parse_sliders(&format!("sliders={}", configs)) {
                        Ok(sliders) => sliders,
                        Err(e) => {
                            eprintln!(
                                "  {}:{} [FAIL] \"{}\" slider parsing error: {:?}",
                                file, line, label, e
                            );
                            failed += 1;
                            continue;
                        }
                    }
                } else {
                    vec![]
                };
                let expr = slider::prepend_slider_bindings(
                    &configs,
                    &vec![0.0; configs.len()],
                    renderer::MarkId::Slider,
                    expr,
                );
                match parser::evaluate(context, expr) {
                    Ok(_) => {
                        println!("  {}:{} [ok] \"{}\"", file, line, label);
                    }
                    Err(e) => {
                        eprintln!(
                            "  {}:{} [FAIL] \"{}\" evaluate error: {:?}",
                            file, line, label, e
                        );
                        failed += 1;
                    }
                }
            }
            Err(errors) => {
                eprintln!(
                    "  {}:{} [FAIL] \"{}\" parse errors: {:?}",
                    file, line, label, errors
                );
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
    let context = load_context();

    let mut total_found = 0;
    let mut total_failed = 0;

    for file in &args.input_files {
        let (found, failed) = check_file(file, &context);
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
