//! File-loading helpers shared between startup (`main`) and the runtime
//! reload effects (`Effect::LoadContext` / `Effect::LoadPrograms`).
//!
//! These functions do file I/O; they're called from `main` to bootstrap
//! and from the effect runner in response to reload requests.

use std::collections::HashMap;
use std::fs;

use crate::builtins;
use crate::parser;
use crate::renderer::{self, MarkId, Program, ProgramSliders, WaveformId};
use crate::slider;

/// Runtime configuration carried in `AppState`. These are the bits of the
/// CLI args that the loader needs to perform a reload — extracted so the
/// effect runner doesn't depend on `clap` / `Args` directly.
#[derive(Debug, Clone)]
pub struct Config {
    pub tempo: u32,
    pub sample_rate: i32,
    pub context_files: Vec<String>,
    pub programs_file: String,
    /// Additional programs supplied via `--program` on the command line.
    /// Reloads re-append these after reading `programs_file`.
    pub additional_programs: Vec<String>,
}

/// Reads and evaluates all configured context files into `context`,
/// replacing whatever was there. Returns a user-visible status message
/// (success count or first error).
pub fn load_context(config: &Config, context: &mut Vec<(String, parser::Expr<MarkId>)>) -> String {
    use parser::Expr;
    context.clear();
    context.push(("tempo".to_string(), Expr::Float(config.tempo as f32)));
    context.push((
        "sample_rate".to_string(),
        Expr::Float(config.sample_rate as f32),
    ));
    builtins::add_prelude(context);
    context.push((
        "mark".to_string(),
        Expr::BuiltIn {
            name: "mark".to_string(),
            function: parser::BuiltInFn(std::rc::Rc::new(renderer::mark)),
        },
    ));

    let mut bindings = 0;
    let mut errors = Vec::new();
    for file in config.context_files.iter() {
        let raw_context = std::fs::read_to_string(file)
            .expect(format!("Failed to read context file: {}", file).as_str());
        // Strip out comments (that is any after // on a line)
        let raw_context: String = raw_context
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
        match parser::parse_context(&raw_context) {
            Ok(parsed_exprs) => {
                println!("Parsed context from {}:", file);
                for (pattern, parsed_expr) in parsed_exprs {
                    match parser::evaluate(context, parsed_expr) {
                        Ok(expr) => {
                            match parser::extend_context(context, &pattern, &expr) {
                                Ok(_) => println!("   {}", &pattern),
                                Err(error) => errors.push(error),
                            }
                            // Not exactly one binding... :shrug:
                            bindings += 1;
                        }
                        Err(error) => {
                            println!(
                                "Error evaluating context expression for {}: {:?}",
                                pattern, error
                            );
                            errors.push(error);
                        }
                    }
                }
            }
            Err(es) => {
                println!("Errors parsing context: {:?}", es);
                errors.extend_from_slice(&es);
            }
        }
    }

    if errors.is_empty() {
        format!("Loaded {} bindings from context", bindings)
    } else {
        format!("Error loading context: {}", errors[0].to_string())
    }
}

/// Reads the configured programs file plus any `additional_programs`,
/// replacing the contents of `programs`. Returns initial slider values
/// (for the slider-update worker) and any parse errors.
pub fn load_programs(
    config: &Config,
    programs: &mut Vec<Program>,
) -> (HashMap<(WaveformId, String), f32>, Vec<parser::Error>) {
    let mut errors = Vec::new();
    programs.clear();
    if !config.programs_file.is_empty() {
        let mut count = 0;
        let contents = fs::read_to_string(&config.programs_file).unwrap_or_default();
        let mut pending_sliders: Option<Vec<parser::Slider>> = None;
        let mut pending_color: Option<(u8, u8, u8)> = None;
        let mut pending_level_db: f32 = 0.0;
        for line in contents.lines() {
            // Check for annotations before stripping comments
            let annos = match parser::parse_annotations(line) {
                Ok(annos) => annos,
                Err(mut e) => {
                    println!("Got errors parsing annotations: {:?}", e);
                    errors.append(&mut e);
                    continue;
                }
            };
            for anno in annos {
                match anno {
                    parser::Annotation::Sliders(sliders) => {
                        pending_sliders = Some(sliders);
                    }
                    parser::Annotation::Color(r, g, b) => {
                        pending_color = Some((r, g, b));
                    }
                    parser::Annotation::Level(v) => {
                        pending_level_db = v;
                    }
                    parser::Annotation::NextBank => {
                        while programs.len() % renderer::PROGRAMS_PER_BANK != 0 {
                            programs.push(Program {
                                text: String::new(),
                                id: renderer::id_from_index(programs.len()),
                                sliders: ProgramSliders::default(),
                                color: None,
                                level_db: 0.0,
                            })
                        }
                    }
                }
            }

            let line = if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
            .trim();
            if !line.is_empty() {
                let sliders = if let Some(configs) = pending_sliders.take() {
                    use parser::SliderFunction;
                    let normalized_values = configs
                        .iter()
                        .map(|c| match &c.function {
                            SliderFunction::Linear {
                                initial_value,
                                min,
                                max,
                            } => ((initial_value - min) / (max - min)).clamp(0.0, 1.0),
                            SliderFunction::UserDefined {
                                normalized_initial_value,
                                ..
                            } => normalized_initial_value.clamp(0.0, 1.0),
                        })
                        .collect();
                    ProgramSliders {
                        configs,
                        normalized_values,
                    }
                } else {
                    ProgramSliders::default()
                };
                let level_db = std::mem::replace(&mut pending_level_db, 0.0);
                programs.push(Program {
                    text: line.to_string(),
                    id: renderer::id_from_index(programs.len()),
                    sliders,
                    color: pending_color.take(),
                    level_db,
                });
                count += 1;
            }
        }
        println!("Loaded {} programs from {}", count, config.programs_file);
    }
    // Add in any additional programs specified on the command line
    for program_text in &config.additional_programs {
        if !program_text.is_empty() {
            programs.push(Program {
                text: program_text.to_string(),
                id: renderer::id_from_index(programs.len()),
                sliders: ProgramSliders::default(),
                color: None,
                level_db: 0.0,
            });
        }
    }
    // Fill up with empty entries if necessary
    while programs.len() < renderer::NUM_PROGRAM_BANKS * renderer::PROGRAMS_PER_BANK {
        programs.push(Program {
            text: String::new(),
            id: renderer::id_from_index(programs.len()),
            sliders: ProgramSliders::default(),
            color: None,
            level_db: 0.0,
        });
    }
    // Copy initial values for each slider for all programs
    let mut last_slider_values: HashMap<(WaveformId, String), f32> = HashMap::new();
    for Program { id, sliders, .. } in programs.iter() {
        for (j, config) in sliders.configs.iter().enumerate() {
            let value =
                slider::denormalize(&config.function, sliders.normalized_values[j]).unwrap_or(0.0);
            last_slider_values.insert(
                (WaveformId::Program(id.clone()), config.label.clone()),
                value,
            );
        }
    }
    (last_slider_values, errors)
}
