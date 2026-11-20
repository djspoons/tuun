//! Which program slot owns each slider a program's waveform reaches. Sliders
//! are implemented as a sort of "optional reader" effect: They are implemented
//! as Marked waveforms, but read their initial value from either a default
//! (provided when they are defined) or from an owner: owners resolve the effect
//! by providing that value.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;

use crate::diagnostics::Source;
use crate::expr;
use crate::ids::{self, MarkId};
use crate::slider;
use crate::slider_effects;
use crate::waveform;

/// A program slot as an owner of sliders: which slot it is, where its text is,
/// and the labels its annotation lists.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerSlot {
    /// The slot's index among its set's programs.
    pub index: usize,
    /// The slot's name in user-visible messages.
    pub name: String,
    /// Where the slot's text is in the backing file; empty when it has no
    /// text there.
    pub text: Range<usize>,
    /// The labels the slot's `sliders` annotation lists.
    pub listed: Vec<String>,
    /// Where that annotation is, when the slot has one.
    pub listing: Option<expr::Span<Source>>,
}

/// The slots that can own the sliders one program's waveform reaches: that
/// program, and the slots whose bindings it is evaluated under.
#[derive(Clone, Debug, PartialEq)]
pub struct Owners {
    /// The program being played.
    pub played: OwnerSlot,
    /// The slots before it in the backing file.
    pub preceding: Vec<OwnerSlot>,
}

impl Owners {
    /// Returns the index of the slot that owns `call`'s slider: the one slot
    /// whose text holds the call or code running around it (see
    /// [`slider::SliderCall::code`]) that lists its label, or the played
    /// program if no such slot does.
    ///
    /// A span names the played program when it is in that program's own text,
    /// and names a slot when it is in that slot's text in the backing file. A
    /// span in a module names no slot. Returns an error at each listing when
    /// two or more such slots list the label.
    pub fn owner(
        &self,
        call: &slider::SliderCall<Source>,
    ) -> Result<usize, Vec<expr::Error<Source>>> {
        self.owner_of(&call.label, call.code())
    }

    /// Returns the index of the slot that owns the slider labeled `label`
    /// called from the code at `code`, the call first and then the code
    /// running around it, as [`Owners::owner`] does.
    pub fn owner_of<'s>(
        &self,
        label: &str,
        code: impl Iterator<Item = &'s expr::Span<Source>>,
    ) -> Result<usize, Vec<expr::Error<Source>>> {
        let mut listing: Vec<&OwnerSlot> = Vec::new();
        for slot in code.filter_map(|span| self.slot(span)) {
            if slot.listed.iter().any(|l| l == label)
                && !listing.iter().any(|s| s.index == slot.index)
            {
                listing.push(slot);
            }
        }
        listing.sort_by_key(|slot| slot.index);
        match listing.as_slice() {
            [] => Ok(self.played.index),
            [one] => Ok(one.index),
            many => {
                let names: Vec<&str> = many.iter().map(|slot| slot.name.as_str()).collect();
                let message = format!(
                    "slider {:?} is listed by {}, and this waveform is built through each of them; \
                     keep only one listing",
                    label,
                    names.join(" and ")
                );
                Err(many
                    .iter()
                    .map(|slot| expr::Error::eval(message.clone(), slot.listing.clone()))
                    .collect())
            }
        }
    }

    /// Returns the slot `span` names, if any.
    fn slot(&self, span: &expr::Span<Source>) -> Option<&OwnerSlot> {
        match span.source {
            Source::Program => Some(&self.played),
            Source::File => std::iter::once(&self.played)
                .chain(&self.preceding)
                .find(|slot| slot.text.contains(&span.range.start)),
            Source::Module(_) => None,
        }
    }

    /// Returns the name of the slot at `index`.
    fn name(&self, index: usize) -> &str {
        std::iter::once(&self.played)
            .chain(&self.preceding)
            .find(|slot| slot.index == index)
            .map_or("", |slot| slot.name.as_str())
    }
}

/// A slider a waveform reaches: the slot that owns it, its label, and its spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Reached {
    pub owner: usize,
    pub label: String,
    pub spec: slider::SliderSpec,
}

/// Rewrites each unresolved slider mark in `waveform` to a
/// [`MarkId::Slider`] of the slot that owns it (see [`Owners::owner`]), and
/// returns the sliders reached, each once, in the order the text reaches
/// their first calls (see [`slider::SliderCall::text_order`]).
///
/// Returns the errors [`Owners::owner`] finds, each once, and an error at
/// each call whose spec differs from that of another call of the same owner
/// and label visited before it.
pub fn resolve(
    waveform: &mut waveform::Waveform<MarkId>,
    owners: &Owners,
) -> Result<Vec<Reached>, Vec<expr::Error<Source>>> {
    // One evaluation of a call can be reached more than once, as through a
    // shared value, so each is resolved once.
    let mut resolved: HashMap<ids::SliderCallRef, Option<usize>> = HashMap::new();
    let mut specs: HashMap<(usize, String), slider::SliderSpec> = HashMap::new();
    // Each slider reached, with the text order of its first call.
    let mut reached: Vec<(Vec<usize>, Reached)> = Vec::new();
    let mut errors: Vec<expr::Error<Source>> = Vec::new();
    waveform::for_each_mark_mut(waveform, &mut |id| {
        let MarkId::SliderCall(call) = id else {
            return;
        };
        let call = call.clone();
        let owner = *resolved
            .entry(call.clone())
            .or_insert_with(|| match owners.owner(&call.0) {
                Ok(owner) => {
                    match specs.entry((owner, call.0.label.clone())) {
                        Entry::Vacant(entry) => {
                            entry.insert(call.0.spec.clone());
                            reached.push((
                                call.0.text_order(),
                                Reached {
                                    owner,
                                    label: call.0.label.clone(),
                                    spec: call.0.spec.clone(),
                                },
                            ));
                        }
                        Entry::Occupied(entry) if *entry.get() != call.0.spec => {
                            errors.push(disagreement(
                                &call.0.label,
                                owners.name(owner),
                                &call.0.spec,
                                entry.get(),
                                call.0.site.clone(),
                            ));
                        }
                        Entry::Occupied(_) => {
                            let order = call.0.text_order();
                            if let Some((first, _)) = reached.iter_mut().find(|(_, slider)| {
                                slider.owner == owner && slider.label == call.0.label
                            }) && order < *first
                            {
                                *first = order;
                            }
                        }
                    }
                    Some(owner)
                }
                Err(found) => {
                    for error in found {
                        if !errors.contains(&error) {
                            errors.push(error);
                        }
                    }
                    None
                }
            });
        if let Some(program) = owner {
            *id = MarkId::Slider {
                program,
                label: call.0.label.clone(),
            };
        }
    });
    if errors.is_empty() {
        reached.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(reached.into_iter().map(|(_, slider)| slider).collect())
    } else {
        Err(errors)
    }
}

/// Returns the error for a call at `site` of the slider labeled `label` of
/// `owner` whose spec, `here`, differs from that of another call, `there`.
fn disagreement(
    label: &str,
    owner: &str,
    here: &slider::SliderSpec,
    there: &slider::SliderSpec,
    site: Option<expr::Span<Source>>,
) -> expr::Error<Source> {
    expr::Error::eval(
        format!(
            "slider {:?} of {} is {} here but {} at another call; \
             calls of one slider must agree",
            label, owner, here, there
        ),
        site,
    )
}

/// Returns the sliders `analysis` finds its program reaching, each once with
/// the slot that owns it, and the errors in how they are owned.
///
/// Those errors are [`Owners::owner`]'s; one at each call whose spec differs
/// from that of an earlier call of the same owner and label; one at each call
/// the played program owns whose spec depends on a parameter; and one at the
/// listing of each other slot that owns a slider reached, when a call of it
/// in that slot's own code has a spec that depends on a parameter, which the
/// slot's callers give. Calls whose label isn't known are skipped.
pub fn check(
    analysis: &slider_effects::Analysis<Source>,
    owners: &Owners,
) -> (Vec<Reached>, Vec<expr::Error<Source>>) {
    let mut reached: Vec<Reached> = Vec::new();
    let mut errors: Vec<expr::Error<Source>> = Vec::new();
    let push = |errors: &mut Vec<expr::Error<Source>>, error| {
        if !errors.contains(&error) {
            errors.push(error);
        }
    };
    // The slots other than the played program that own a slider reached.
    let mut listed: Vec<(usize, &str)> = Vec::new();
    for call in &analysis.program {
        let Ok(label) = &call.label else { continue };
        let owner = match owners.owner_of(label, call.code()) {
            Ok(owner) => owner,
            Err(found) => {
                for error in found {
                    push(&mut errors, error);
                }
                continue;
            }
        };
        match &call.spec {
            Ok(spec) => match reached
                .iter()
                .find(|slider| slider.owner == owner && slider.label == *label)
            {
                Some(slider) if slider.spec != *spec => {
                    let error = disagreement(
                        label,
                        owners.name(owner),
                        spec,
                        &slider.spec,
                        call.site.clone(),
                    );
                    push(&mut errors, error);
                }
                Some(_) => {}
                None => reached.push(Reached {
                    owner,
                    label: label.clone(),
                    spec: spec.clone(),
                }),
            },
            Err(slider_effects::Open {
                part,
                parameter: Some(parameter),
            }) if owner == owners.played.index => {
                let error = expr::Error::eval(
                    format!(
                        "slider {:?}'s {} depends on `{}`, which isn't known until it plays",
                        label, part, parameter.name
                    ),
                    call.site.clone(),
                );
                push(&mut errors, error);
            }
            Err(_) => {}
        }
        if owner != owners.played.index && !listed.contains(&(owner, label.as_str())) {
            listed.push((owner, label));
        }
    }
    for (owner, label) in listed {
        let Some(slot) = owners.preceding.iter().find(|slot| slot.index == owner) else {
            continue;
        };
        let open = analysis
            .definitions
            .iter()
            .find_map(|call| match &call.spec {
                Err(slider_effects::Open {
                    part,
                    parameter: Some(parameter),
                }) if call.label.as_deref() == Ok(label)
                    && owners.owner_of(label, call.code()) == Ok(owner) =>
                {
                    Some((part, parameter))
                }
                _ => None,
            });
        if let Some((part, parameter)) = open {
            let error = expr::Error::eval(
                format!(
                    "{} lists slider {:?}, but its {} depends on `{}`, which its callers give; \
                     list it where `{}` is given instead",
                    slot.name, label, part, parameter.name, parameter.name
                ),
                slot.listing.clone(),
            );
            push(&mut errors, error);
        }
    }
    (reached, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::eval::CallPath;
    use crate::slider::Curve;
    use crate::waveform::BinaryOperator;
    use crate::waveform::Waveform::{BinaryOp, Const, Marked};

    /// Returns a file span over `range`.
    fn file(range: Range<usize>) -> expr::Span<Source> {
        expr::Span::new(Source::File, range)
    }

    /// Returns a slot at `index` whose text is `text` in the file, listing
    /// `listed` in an annotation at the slot's first byte.
    fn slot(index: usize, text: Range<usize>, listed: &[&str]) -> OwnerSlot {
        OwnerSlot {
            index,
            name: format!("S{index}"),
            listing: (!listed.is_empty()).then(|| file(text.start..text.start + 1)),
            text,
            listed: listed.iter().map(|label| label.to_string()).collect(),
        }
    }

    /// Returns owners where slot 1 (file text 50..70) is played and slot 0
    /// (file text 10..40) precedes it, with each listing the given labels.
    fn owners(preceding: &[&str], played: &[&str]) -> Owners {
        Owners {
            played: slot(1, 50..70, played),
            preceding: vec![slot(0, 10..40, preceding)],
        }
    }

    /// Returns a call of the slider `label` written at `site`, inside code
    /// written at `path`, outermost first.
    fn call(
        label: &str,
        default: f32,
        site: expr::Span<Source>,
        path: &[expr::Span<Source>],
    ) -> slider::SliderCall<Source> {
        slider::SliderCall {
            label: label.to_string(),
            spec: slider::SliderSpec::new(default, 0.0, 1.0, Curve::Linear).unwrap(),
            site: Some(site),
            path: path.iter().fold(CallPath::empty(), |path, code| {
                path.enter(Some(code.clone()), None)
            }),
        }
    }

    /// A slider call written in a module, inside a function in slot 0's text,
    /// inside the played program's own text.
    fn through_slot_0(label: &str) -> slider::SliderCall<Source> {
        call(
            label,
            0.5,
            expr::Span::new(Source::Module(0), 0..5),
            &[expr::Span::new(Source::Program, 0..3), file(20..25)],
        )
    }

    #[test]
    fn a_slider_no_slot_lists_is_the_played_programs() {
        assert_eq!(owners(&[], &[]).owner(&through_slot_0("q")), Ok(1));
    }

    #[test]
    fn a_slider_is_the_listing_slots_on_its_path() {
        assert_eq!(owners(&["q"], &[]).owner(&through_slot_0("q")), Ok(0));
        assert_eq!(owners(&[], &["q"]).owner(&through_slot_0("q")), Ok(1));
        // Another label's listing doesn't count.
        assert_eq!(owners(&["r"], &[]).owner(&through_slot_0("q")), Ok(1));
    }

    #[test]
    fn a_listing_slot_off_the_path_owns_nothing() {
        let in_program = call("q", 0.5, expr::Span::new(Source::Program, 4..9), &[]);
        assert_eq!(owners(&["q"], &[]).owner(&in_program), Ok(1));
    }

    #[test]
    fn a_span_names_the_slot_whose_file_text_holds_it() {
        // The played program's file text names it too.
        let in_played = call("q", 0.5, file(55..60), &[]);
        assert_eq!(owners(&["q"], &["q"]).owner(&in_played), Ok(1));
        // A module span names no slot, even at slot 0's offsets.
        let in_module = call("q", 0.5, expr::Span::new(Source::Module(0), 20..25), &[]);
        assert_eq!(owners(&["q"], &[]).owner(&in_module), Ok(1));
    }

    #[test]
    fn two_listing_slots_on_a_path_are_an_error_at_each_listing() {
        let errors = owners(&["q"], &["q"])
            .owner(&through_slot_0("q"))
            .unwrap_err();
        let message = "slider \"q\" is listed by S0 and S1, and this waveform is built through \
                       each of them; keep only one listing";
        assert_eq!(
            errors,
            [
                expr::Error::eval(message, Some(file(10..11))),
                expr::Error::eval(message, Some(file(50..51))),
            ]
        );
    }

    /// Returns `call` as a marked constant.
    fn marked(call: &Arc<slider::SliderCall<Source>>) -> Box<waveform::Waveform<MarkId>> {
        Box::new(Marked {
            id: MarkId::SliderCall(ids::SliderCallRef(call.clone())),
            waveform: Box::new(Const(call.spec.default)),
        })
    }

    /// Returns the sum of `a` and `b`, each marked.
    fn sum(
        a: &Arc<slider::SliderCall<Source>>,
        b: &Arc<slider::SliderCall<Source>>,
    ) -> waveform::Waveform<MarkId> {
        BinaryOp(BinaryOperator::Add, marked(a), marked(b))
    }

    /// Returns the slider mark of `label` owned by `program`.
    fn owned(program: usize, label: &str) -> MarkId {
        MarkId::Slider {
            program,
            label: label.to_string(),
        }
    }

    #[test]
    fn resolving_names_each_marks_owner() {
        let a = Arc::new(through_slot_0("q"));
        let b = Arc::new(call("r", 0.5, expr::Span::new(Source::Program, 4..9), &[]));
        let mut w = sum(&a, &b);
        assert!(resolve(&mut w, &owners(&["q"], &[])).is_ok());
        let mut ids = Vec::new();
        waveform::for_each_mark_mut(&mut w, &mut |id| ids.push(id.clone()));
        assert_eq!(ids, [owned(0, "q"), owned(1, "r")]);
    }

    #[test]
    fn resolving_returns_each_slider_reached_once_in_text_order() {
        let r = Arc::new(call("r", 0.5, expr::Span::new(Source::Program, 4..9), &[]));
        let q = Arc::new(through_slot_0("q"));
        let again = Arc::new(call(
            "r",
            0.5,
            expr::Span::new(Source::Program, 12..17),
            &[],
        ));
        let mut w = BinaryOp(BinaryOperator::Add, Box::new(sum(&r, &q)), marked(&again));
        let reached = resolve(&mut w, &owners(&["q"], &[])).unwrap();
        let spec = slider::SliderSpec::new(0.5, 0.0, 1.0, Curve::Linear).unwrap();
        // `r` is visited first, but `q` is reached through a call at offset
        // 0 of the program's text, before `r`'s at 4.
        assert_eq!(
            reached,
            [
                Reached {
                    owner: 0,
                    label: "q".to_string(),
                    spec: spec.clone()
                },
                Reached {
                    owner: 1,
                    label: "r".to_string(),
                    spec
                },
            ]
        );
    }

    #[test]
    fn calls_of_one_slider_must_agree() {
        let first = Arc::new(call("q", 0.5, expr::Span::new(Source::Program, 0..5), &[]));
        let second = Arc::new(call(
            "q",
            0.25,
            expr::Span::new(Source::Program, 8..13),
            &[],
        ));
        let errors = resolve(&mut sum(&first, &second), &owners(&[], &[])).unwrap_err();
        assert_eq!(
            errors,
            [expr::Error::eval(
                "slider \"q\" of S1 is 0.25 (0 to 1, linear) here but 0.5 (0 to 1, linear) at \
                 another call; calls of one slider must agree",
                Some(expr::Span::new(Source::Program, 8..13)),
            )]
        );
        // Calls of sliders with different owners need not agree.
        let first = Arc::new(through_slot_0("q"));
        assert!(resolve(&mut sum(&first, &second), &owners(&["q"], &[])).is_ok());
    }

    #[test]
    fn a_call_reached_twice_is_reported_once() {
        let a = Arc::new(through_slot_0("q"));
        let errors = resolve(&mut sum(&a, &a), &owners(&["q"], &["q"])).unwrap_err();
        assert_eq!(errors.len(), 2, "{errors:?}");
    }
}
