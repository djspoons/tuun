//! Embedded built-in tuun modules and helpers for the `open` syntax.
//!
//! [`EMBEDDED_MODULES`] is the single source of truth for which modules
//! are bundled into binaries that don't have filesystem access (the wasm
//! bindings and the doc-example checker). Adding a new module means
//! adding one entry here.

/// `(dotted_module_path, source_contents)` pairs. The dotted path is
/// the same shape that `Binding::Open(path)` carries; running
/// `parser::parse_file` on the contents yields the module's bindings.
pub const EMBEDDED_MODULES: &[(&str, &str)] = &[("std", include_str!("../../lib/v0/std.tuun"))];

/// Parses a JSON array of dotted module paths like `["std", "foo.bar"]`
/// into the path-component vectors that `Binding::Open` expects. `"[]"`
/// (or whitespace around it) yields an empty list. Both `"…"` and
/// `'…'` quoting are accepted since HTML attributes commonly use
/// single quotes.
pub fn parse_open_json(json: &str) -> Result<Vec<Vec<String>>, String> {
    let trimmed = json.trim();
    if trimmed == "[]" {
        return Ok(Vec::new());
    }
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("JSON must be an array".to_string());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut result = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let unquoted = item
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| item.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .ok_or_else(|| format!("Invalid module path entry: {}", item))?;
        if unquoted.is_empty() {
            return Err("Module path entries must be non-empty".to_string());
        }
        let parts: Vec<String> = unquoted.split('.').map(|s| s.to_string()).collect();
        result.push(parts);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_open_json() {
        assert_eq!(parse_open_json("[]").unwrap(), Vec::<Vec<String>>::new());
        assert_eq!(
            parse_open_json(r#"["std"]"#).unwrap(),
            vec![vec!["std".to_string()]]
        );
        assert_eq!(
            parse_open_json(r#"["std", "foo.bar"]"#).unwrap(),
            vec![
                vec!["std".to_string()],
                vec!["foo".to_string(), "bar".to_string()]
            ]
        );
        // Single-quoted entries (HTML attributes often use them).
        assert_eq!(
            parse_open_json("['std', 'foo.bar']").unwrap(),
            vec![
                vec!["std".to_string()],
                vec!["foo".to_string(), "bar".to_string()]
            ]
        );
        // Not an array.
        assert!(parse_open_json(r#""std""#).is_err());
        // Unquoted entry.
        assert!(parse_open_json("[std]").is_err());
    }
}
