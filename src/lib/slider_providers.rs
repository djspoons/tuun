//! The program slots that can provide the sliders one program's waveform
//! reaches. Sliders are a sort of reader effect: a slider's value is read from
//! its provider, the listing binding whose code is running where the slider
//! is read, or else the program being played.

use std::ops::Range;
use std::sync::Arc;

use crate::diagnostics::Source;
use crate::expr;
use crate::ids::MarkId;
use crate::slider;
use crate::waveform;

/// A program slot as a provider of sliders: which slot it is, where its text
/// is, and the labels its annotation lists.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderSlot {
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

/// The slots that can provide the sliders one program's waveform reaches: that
/// program, and the slots whose bindings it is evaluated under.
#[derive(Clone, Debug, PartialEq)]
pub struct Providers {
    /// The program being played.
    pub played: ProviderSlot,
    /// The slots before it in the backing file.
    pub preceding: Vec<ProviderSlot>,
}

impl Providers {
    /// Returns the slot whose text holds the code written at `span`, if any.
    ///
    /// A span is in the played program's text when it is in that program's
    /// own text, and in a slot's text when it is in that slot's text in the
    /// backing file. A span in a module is in no slot's text.
    pub fn slot(&self, span: &expr::Span<Source>) -> Option<&ProviderSlot> {
        slot(&self.played, &self.preceding, span)
    }

    /// Returns the slot at `index`, if it is one of these.
    pub fn slot_at(&self, index: usize) -> Option<&ProviderSlot> {
        std::iter::once(&self.played)
            .chain(&self.preceding)
            .find(|slot| slot.index == index)
    }

    /// Returns the listings among these slots: each slot that lists sliders,
    /// with its index as its id, found for the code written in its text (see
    /// [`Providers::slot`]).
    ///
    /// # Example
    ///
    /// ```
    /// use tuun::diagnostics::Source;
    /// use tuun::expr::Span;
    /// use tuun::slider_providers::{ProviderSlot, Providers};
    ///
    /// let slot = |index: usize, text: std::ops::Range<usize>, listed: &[&str]| ProviderSlot {
    ///     index,
    ///     name: format!("S{index}"),
    ///     text,
    ///     listed: listed.iter().map(|label| label.to_string()).collect(),
    ///     listing: None,
    /// };
    /// let providers = Providers {
    ///     played: slot(1, 50..70, &[]),
    ///     preceding: vec![slot(0, 10..40, &["mix"])],
    /// };
    /// let listings = providers.listings();
    /// let found = listings.find(&Span::new(Source::File, 20..25)).unwrap();
    /// assert_eq!((found.id, &found.labels[..]), (0, &["mix".to_string()][..]));
    /// // The played program lists nothing.
    /// assert_eq!(listings.find(&Span::new(Source::Program, 0..5)), None);
    /// ```
    pub fn listings(&self) -> slider::Listings<Source> {
        let played = self.played.clone();
        let preceding = self.preceding.clone();
        slider::Listings::new(move |span| {
            let slot = slot(&played, &preceding, span)?;
            if slot.listed.is_empty() {
                return None;
            }
            Some(slider::Listing {
                id: slot.index,
                labels: Arc::from(slot.listed.clone()),
            })
        })
    }
}

/// Returns the slot among `played` and `preceding` whose text holds the code
/// written at `span`, as [`Providers::slot`] does.
fn slot<'s>(
    played: &'s ProviderSlot,
    preceding: &'s [ProviderSlot],
    span: &expr::Span<Source>,
) -> Option<&'s ProviderSlot> {
    match span.source {
        Source::Program => Some(played),
        Source::File => std::iter::once(played)
            .chain(preceding)
            .find(|slot| slot.text.contains(&span.range.start)),
        Source::Module(_) => None,
    }
}

/// A slider a program can reach: the slot that provides it, its label, and its
/// spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Reached {
    pub provider: usize,
    pub label: String,
    pub spec: slider::SliderSpec,
}

/// Makes `waveform` the program at `program`'s: each slider in it that the
/// played program provides becomes that program's.
pub fn provide(waveform: &mut waveform::Waveform<MarkId>, program: usize) {
    waveform::for_each_mark_mut(waveform, &mut |id| {
        if let MarkId::PlayedSlider { label } = id {
            *id = MarkId::Slider {
                program,
                label: std::mem::take(label),
            };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::BinaryOperator;
    use crate::waveform::Waveform::{BinaryOp, Const, Marked};

    /// Returns a slot at `index` whose text is `text` in the file, listing
    /// `listed`.
    fn slot(index: usize, text: Range<usize>, listed: &[&str]) -> ProviderSlot {
        ProviderSlot {
            index,
            name: format!("S{index}"),
            text,
            listed: listed.iter().map(|label| label.to_string()).collect(),
            listing: None,
        }
    }

    /// Returns providers where slot 1 (file text 50..70) is played and slot 0
    /// (file text 10..40) precedes it, with each listing the given labels.
    fn providers(preceding: &[&str], played: &[&str]) -> Providers {
        Providers {
            played: slot(1, 50..70, played),
            preceding: vec![slot(0, 10..40, preceding)],
        }
    }

    /// Returns the id of the listing found for code written at `span`.
    fn found(providers: &Providers, source: Source, range: Range<usize>) -> Option<usize> {
        providers
            .listings()
            .find(&expr::Span::new(source, range))
            .map(|listing| listing.id)
    }

    #[test]
    fn code_in_a_listing_slots_text_is_in_that_listing() {
        let providers = providers(&["q"], &["r"]);
        assert_eq!(found(&providers, Source::File, 20..25), Some(0));
        // The played program's own text and its text in the file.
        assert_eq!(found(&providers, Source::Program, 0..3), Some(1));
        assert_eq!(found(&providers, Source::File, 55..60), Some(1));
    }

    #[test]
    fn code_elsewhere_is_in_no_listing() {
        let providers = providers(&["q"], &[]);
        // A slot that lists nothing is no listing.
        assert_eq!(found(&providers, Source::Program, 0..3), None);
        // Text between slots, and a module at a slot's offsets.
        assert_eq!(found(&providers, Source::File, 42..45), None);
        assert_eq!(found(&providers, Source::Module(0), 20..25), None);
    }

    #[test]
    fn providing_makes_the_played_programs_sliders_a_programs() {
        let marked = |id: MarkId| {
            Box::new(Marked {
                id,
                waveform: Box::new(Const(0.5)),
            })
        };
        let mut w = BinaryOp(
            BinaryOperator::Add,
            marked(MarkId::PlayedSlider {
                label: "q".to_string(),
            }),
            marked(MarkId::Slider {
                program: 0,
                label: "r".to_string(),
            }),
        );
        provide(&mut w, 3);
        let mut ids = Vec::new();
        waveform::for_each_mark_mut(&mut w, &mut |id| ids.push(id.clone()));
        let slider = |program, label: &str| MarkId::Slider {
            program,
            label: label.to_string(),
        };
        assert_eq!(ids, [slider(3, "q"), slider(0, "r")]);
    }
}
