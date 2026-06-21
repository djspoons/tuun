use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path;
use std::sync::mpsc;
use std::time;

use crate::builtins;
use crate::optimizer;
use crate::parser;
use crate::renderer;
use crate::renderer::{MarkId, Program, WaveformId};
use crate::slider;
use crate::tracker;
use crate::waveform;

pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// One slot in [`PlayHelper`]'s module cache: the file's mtime at the time
/// we parsed it, plus a leaked `&'static` slice of the parsed bindings.
// See the field doc on `PlayHelper::modules` for the leak strategy.
struct ModuleCacheEntry {
    mtime: time::SystemTime,
    bindings: &'static [parser::SourceBinding<MarkId>],
}

pub struct PlayHelper {
    tempo: u32,
    beats_per_measure: u32,

    /// Built-ins + environment-derived definitions; implicitly opened at
    /// the top of every other loaded module — see [`PlayHelper::resolve`].
    prelude: Vec<parser::SourceBinding<MarkId>>,
    /// Filesystem root for module resolution. A module path
    /// `["foo", "bar"]` is looked up as `<library_root>/foo/bar.tuun`.
    library_root: path::PathBuf,
    /// Cache of loaded modules, keyed by module path.
    ///
    /// Each entry pairs the file's last-seen mtime with a leaked `&'static`
    /// slice of bindings. On `resolve`, we stat the file and re-read+re-parse
    /// if the mtime has changed; otherwise we hand back the cached slice.
    ///
    /// Because previously-returned borrows must stay valid for the duration of
    /// an evaluate session, invalidation does NOT free the old allocation — the
    /// stale leak stays in memory until process exit. That's bounded by `(files
    /// loaded) * (edits per file)`, which is fine for tuun's interactive
    /// workflow.
    modules: RefCell<HashMap<Vec<String>, ModuleCacheEntry>>,
    precomputing_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    fast_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
}

impl PlayHelper {
    pub fn new(
        sample_rate: u32,
        tempo: u32,
        beats_per_measure: u32,
        library_root: path::PathBuf,
        precomputing_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        fast_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    ) -> PlayHelper {
        // Construct the prelude from the built-ins and any environment specific bindings.
        let mut prelude = Vec::new();
        builtins::add_bindings(&mut prelude);

        use parser::SourceExpr;
        fn def<M>(id: &str, expr: SourceExpr<M>) -> parser::SourceBinding<M> {
            use parser::Binding;
            use parser::Pattern;
            Binding::Definition(Pattern::Identifier(id.to_string()), expr).into()
        }
        prelude.push(def("tempo", SourceExpr::float(tempo as f32)));
        prelude.push(def("sample_rate", SourceExpr::float(sample_rate as f32)));

        prelude.push(def(
            "mark",
            SourceExpr::from(parser::Expr::BuiltIn {
                name: "mark".to_string(),
                function: parser::BuiltInFn(std::rc::Rc::new(renderer::mark)),
            }),
        ));

        PlayHelper {
            tempo,
            beats_per_measure,
            prelude,
            library_root,
            modules: RefCell::new(HashMap::new()),
            precomputing_command_sender,
            fast_command_sender,
        }
    }

    /// Resolves a module path to its (parsed) bindings.
    ///
    /// The special path `["__prelude"]` returns the in-memory prelude built by
    /// [`PlayHelper::new`]. Other paths are mapped to a file under
    /// [`library_root`](Self::library_root): `["foo", "bar"]` →
    /// `<library_root>/foo/bar.tuun`.
    ///
    /// Each call stats the file and compares its mtime with the cached entry's.
    /// On a match we hand back the existing slice; on a mismatch or miss, we
    /// re-read, re-parse, prepend an implicit `open __prelude`, leak the new
    /// bindings, and replace the cache entry. The previous leaked allocation
    /// stays in memory — any borrows handed out before invalidation remain
    /// valid (essential for recursive resolve calls during one `evaluate`
    /// session).
    fn resolve(&self, path: &[String]) -> Result<&[parser::SourceBinding<MarkId>], parser::Error> {
        if path.len() == 1 && path[0] == "__prelude" {
            return Ok(&self.prelude);
        }

        // Build the file path: <library_root>/<path components>.tuun.
        let mut file_path = self.library_root.clone();
        for part in path {
            file_path.push(part);
        }
        file_path.set_extension("tuun");

        // Stat the file once. If the cache has a matching mtime, return the
        // cached slice without re-reading or re-parsing.
        let current_mtime = fs::metadata(&file_path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                parser::Error::new(format!(
                    "Failed to stat module {}: {}",
                    file_path.display(),
                    e
                ))
            })?;
        if let Some(entry) = self.modules.borrow().get(path)
            && entry.mtime == current_mtime
        {
            return Ok(entry.bindings);
        }

        // Cache miss or stale — reload.
        let contents = fs::read_to_string(&file_path).map_err(|e| {
            parser::Error::new(format!(
                "Failed to read module {}: {}",
                file_path.display(),
                e
            ))
        })?;
        // TODO these two error cases are a little hard to unravel, but :shrug:
        let (mut bindings, errors) =
            parser::parse_module::<MarkId>(&contents).map_err(|errors| {
                errors.into_iter().next().unwrap_or_else(|| {
                    parser::Error::new(format!("Parse failed for {}", file_path.display()))
                })
            })?;
        if !errors.is_empty() {
            errors.into_iter().next().unwrap_or_else(|| {
                return parser::Error::new(format!("Parse failed for {}", file_path.display()));
            });
        }

        // Every loaded module implicitly opens the prelude as its first binding
        // so it can reference prelude names without an explicit `open __prelude`
        // line.
        bindings.insert(
            0,
            parser::Binding::Open(vec!["__prelude".to_string()]).into(),
        );

        // Leak the parsed bindings so we can hand out a stable
        // `&[SourceBinding]` reference. On a stale-cache reload, the previous
        // leak isn't freed — any borrows from before the reload (e.g. earlier
        // in the same evaluate session) remain valid.
        // TODO consider other ways of caching that don't depend on leak
        let leaked: &'static [parser::SourceBinding<MarkId>] =
            Box::leak(bindings.into_boxed_slice());
        self.modules.borrow_mut().insert(
            path.to_vec(),
            ModuleCacheEntry {
                mtime: current_mtime,
                bindings: leaked,
            },
        );
        Ok(leaked)
    }

    /// Evaluates a program in preparation for playback or for installing as the
    /// keys instrument. Returns an expression if the program parses and
    /// evaluates successfully; otherwise returns a message.
    ///
    /// The program is evaluated in the context of the file-level bindings
    /// preceding it (per its `binding_index`) plus its sliders. Bindings after
    /// it are ignored.
    pub fn evaluate_program(
        &mut self,
        file_bindings: &[parser::SourceBinding<MarkId>],
        program: &Program,
    ) -> Result<parser::SourceExpr<MarkId>, String> {
        let expr = match parser::parse_program(&program.text) {
            Err(errors) => {
                println!("Errors while parsing input: {:?}", errors);
                let message = format!("Error: {}", errors[0]);
                return Err(message);
            }
            Ok(expr) => expr,
        };
        println!("parser::parse_program returned: {}", &expr);

        let mut bindings: Vec<parser::SourceBinding<MarkId>> =
            file_bindings[..program.binding_index].to_vec();
        // TODO this is a pretty big hack but there's an interesting question
        // about what sliders in *other* bindings mean. To avoid answering that
        // for the moment, just assume that only "_" bindings have sliders and
        // that they can't be used so we can safely filter them out here. I
        // think the right answer is that we should bind sliders uniquely in
        // each binding... or at least those that have slots? (Otherwise, how
        // can you modify the slider value?)
        bindings.retain(|b| match &b.binding {
            parser::Binding::Definition(p, _) => {
                !matches!(p, parser::Pattern::Identifier(v) if v == "_")
            }
            _ => true,
        });
        slider::append_slider_bindings(
            &program.sliders.configs,
            &program.sliders.normalized_values,
            MarkId::Slider,
            &mut bindings,
        );

        let expr = match parser::evaluate(|path| self.resolve(path), &bindings, expr) {
            Err(error) => {
                println!("Errors while evaluating input: {:?}", error);
                return Err(format!("Error: {}", error));
            }
            Ok(expr) => expr,
        };
        println!("parser::evaluate returned: {}", &expr);
        Ok(expr)
    }

    /// Plays a program as a waveform. Returns the user-visible message on
    /// success or failure.
    pub fn play_program_as_waveform(
        &mut self,
        program: &Program,
        status: &tracker::Status<WaveformId, MarkId>,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    ) -> Option<String> {
        let message;
        let repeat_every;
        if let Some(measures) = repeat_after_measures {
            let beats = (measures * self.beats_per_measure) as u64;
            message = format!("Looping waveform {} every {:?} beats", program.id, beats);
            repeat_every = Some(duration_from_beats(self.tempo, beats));
        } else {
            // Otherwise, play it once
            message = format!("Playing waveform {}", program.id);
            repeat_every = None;
        }
        let start = if start_at_next_measure {
            Some(next_measure_start(status))
        } else {
            None
        };
        if let Some(waveform) = program.cached_waveform.clone() {
            let waveform = optimizer::optimize(waveform);
            println!("optimizer::optimize returned: {}", &waveform);
            if start_at_next_measure {
                &mut self.precomputing_command_sender
            } else {
                &mut self.fast_command_sender
            }
            .send(tracker::Command::Play {
                // TODO maybe extend the top-level mark to the full measure?
                id: WaveformId::Program(program.id),
                waveform: build_top_level_waveform(waveform, program.level_db),
                start,
                repeat_every,
            })
            .unwrap();
            Some(message)
        } else {
            None
        }
    }

    pub fn stop_waveform(&mut self, id: WaveformId) {
        use waveform::{Operator, Waveform::*};
        const STOP_DURATION_SECS: f32 = 0.05;
        self.fast_command_sender
            .send(tracker::Command::Modify {
                id,
                mark_id: MarkId::Terminator,
                waveform: Fin {
                    length: Box::new(BinaryPointOp(
                        Operator::Subtract,
                        Box::new(Time(())),
                        Box::new(Const(STOP_DURATION_SECS)),
                    )),
                    waveform: Box::new(BinaryPointOp(
                        Operator::Subtract,
                        Box::new(Const(1.0)),
                        Box::new(BinaryPointOp(
                            Operator::Multiply,
                            Box::new(Time(())),
                            Box::new(Const(1.0 / STOP_DURATION_SECS)),
                        )),
                    )),
                },
            })
            .unwrap();
    }

    pub fn start_beats(
        &self,
        status_receiver: &mpsc::Receiver<tracker::Status<WaveformId, MarkId>>,
    ) {
        // Play the odd Beats waveform starting immediately and repeating every two measures
        self.precomputing_command_sender
            .send(tracker::Command::Play {
                id: WaveformId::Beats(false),
                waveform: self.beats_waveform(),
                start: None,
                repeat_every: Some(
                    duration_from_beats(self.tempo, self.beats_per_measure as u64) * 2,
                ),
            })
            .unwrap();
        // We need to wait to start the even Beats until we know when the odd Beats started
        'start_even_beats: loop {
            if let Ok(status) = status_receiver.recv() {
                for mark in status.marks {
                    if mark.waveform_id == WaveformId::Beats(false)
                        && mark.mark_id == MarkId::TopLevel
                    {
                        self.precomputing_command_sender
                            .send(tracker::Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: self.beats_waveform(),
                                start: Some(mark.start + mark.duration),
                                repeat_every: Some(
                                    duration_from_beats(self.tempo, self.beats_per_measure as u64)
                                        * 2,
                                ),
                            })
                            .unwrap();
                        break 'start_even_beats;
                    }
                }
            }
        }
    }
}

pub fn build_top_level_waveform(
    waveform: waveform::Waveform<MarkId>,
    level_db: f32,
) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryPointOp, Const, Marked};
    Marked {
        id: MarkId::TopLevel,
        waveform: Box::new(BinaryPointOp(
            waveform::Operator::Multiply,
            Box::new(BinaryPointOp(
                waveform::Operator::Multiply,
                Box::new(waveform),
                Box::new(Marked {
                    id: MarkId::Amplitude,
                    waveform: Box::new(Const(db_to_amplitude(level_db))),
                }),
            )),
            Box::new(Marked {
                id: MarkId::Terminator,
                waveform: Box::new(Const(1.0)),
            }),
        )),
    }
}

// Returns the start time of the next measure
fn next_measure_start(status: &tracker::Status<WaveformId, MarkId>) -> time::Instant {
    for mark in &status.marks {
        match mark.waveform_id {
            WaveformId::Beats(_)
                if mark.mark_id == MarkId::TopLevel && mark.start > time::Instant::now() =>
            {
                return mark.start;
            }
            _ => (),
        }
    }
    panic!("No next measure found in marks");
}

fn duration_from_beats(tempo: u32, beats: u64) -> time::Duration {
    time::Duration::from_secs_f32(beats as f32 * 60.0 / tempo as f32)
}

impl PlayHelper {
    /// Builds the per-measure beats waveform — a sequence of `mark`-tagged
    /// short silences, one per beat — used to keep timing visible to the
    /// rest of the runtime.
    pub fn beats_waveform(&self) -> waveform::Waveform<MarkId> {
        let seconds_per_beat = duration_from_beats(self.tempo, 1);
        let mut ws = Vec::new();
        for i in 0..self.beats_per_measure {
            ws.push(format!(
                "0 | fin(time - {}) | seq(time - {}) | mark({})",
                seconds_per_beat.as_secs_f32(),
                seconds_per_beat.as_secs_f32(),
                i + 1
            ));
        }
        let expr = match parser::parse_program(&format!("<[{}]>", ws.join(", "))) {
            Ok(expr) => expr,
            Err(errors) => panic!("Error parsing beats waveform: {:?}", errors),
        };
        let bindings: Vec<parser::SourceBinding<MarkId>> =
            vec![parser::Binding::Open(vec!["__prelude".to_string()]).into()];
        match parser::evaluate(|path| self.resolve(path), &bindings, expr).map(|s| s.expr) {
            Ok(parser::Expr::Seq { waveform, .. }) => match waveform.expr {
                parser::Expr::Waveform(waveform) => waveform::Waveform::<MarkId>::Marked {
                    id: MarkId::TopLevel,
                    waveform: Box::new(optimizer::optimize(waveform)),
                },
                expr => panic!("Error creating beats waveform with seq, got {}", expr),
            },
            Ok(expr) => panic!("Error creating beats waveform, got {}", expr),
            Err(errors) => panic!("Error evaluating beats waveform: {:?}", errors),
        }
    }
}
