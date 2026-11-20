//! The program slots of a file that can provide sliders. Sliders are a sort of
//! reader effect: a slider's value is read from its provider, the listing
//! binding whose code is running where the slider is read, or else the program
//! being played.

use std::ops::Range;
use std::sync::Arc;

use crate::diagnostics::Source;
use crate::expr;
use crate::ids::MarkId;
use crate::slider;
use crate::slider_analysis::Reached;
use crate::waveform;

/// A program slot as a provider of sliders: its name, where its text is, and
/// the sliders its annotation lists.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderSlot {
    /// The slot's name in user-visible messages.
    pub name: String,
    /// Where the slot's text is in the backing file; empty when it has no
    /// text there.
    pub text: Range<usize>,
    /// The entries of the slot's `sliders` annotation: each label it lists,
    /// with the value it saves, if any.
    pub listed: Vec<expr::Slider>,
    /// Where that annotation is, when the slot has one.
    pub listing: Option<expr::Span<Source>>,
}

/// The program slots of a file, each of which can provide sliders, by index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Providers {
    pub slots: Vec<ProviderSlot>,
}

impl Providers {
    /// Returns the index of the slot whose text holds the code written at
    /// `span`, when the program at `played` is played, if any.
    ///
    /// A span is in the played program's text when it is in that program's
    /// own text, and in a slot's text when it is in that slot's text in the
    /// backing file. A span in a module is in no slot's text.
    pub fn slot(&self, played: usize, span: &expr::Span<Source>) -> Option<usize> {
        match span.source {
            Source::Program => Some(played),
            Source::File => self
                .slots
                .iter()
                .position(|slot| slot.text.contains(&span.range.start)),
            Source::Module(_) => None,
        }
    }

    /// Returns the listings among these slots when the program at `played` is
    /// played: each slot that lists sliders, with its index as its id, found
    /// for the code written in its text (see [`Providers::slot`]).
    ///
    /// # Example
    ///
    /// ```
    /// use tuun::diagnostics::Source;
    /// use tuun::expr::{Slider, Span};
    /// use tuun::slider_providers::{ProviderSlot, Providers};
    ///
    /// let slot = |text: std::ops::Range<usize>, listed: &[&str]| ProviderSlot {
    ///     name: "slot".to_string(),
    ///     text,
    ///     listed: listed
    ///         .iter()
    ///         .map(|label| Slider { label: label.to_string(), value: None })
    ///         .collect(),
    ///     listing: None,
    /// };
    /// let providers = Providers {
    ///     slots: vec![slot(10..40, &["mix"]), slot(50..70, &[])],
    /// };
    /// let listings = providers.listings(1);
    /// let found = listings.find(&Span::new(Source::File, 20..25)).unwrap();
    /// assert_eq!((found.id, &found.labels[..]), (0, &["mix".to_string()][..]));
    /// // The played program lists nothing.
    /// assert_eq!(listings.find(&Span::new(Source::Program, 0..5)), None);
    /// ```
    pub fn listings(&self, played: usize) -> slider::Listings<Source> {
        let providers = self.clone();
        // Each slot's listing, built once rather than for each span found.
        let listings: Vec<Option<slider::Listing>> = self
            .slots
            .iter()
            .enumerate()
            .map(|(id, slot)| {
                let labels: Vec<String> = slot
                    .listed
                    .iter()
                    .map(|entry| entry.label.clone())
                    .collect();
                (!labels.is_empty()).then(|| slider::Listing {
                    id,
                    labels: Arc::from(labels),
                })
            })
            .collect();
        slider::Listings::new(move |span| {
            let id = providers.slot(played, span)?;
            listings.get(id)?.clone()
        })
    }

    /// Returns the warnings about the sliders a program can reach,
    /// `reached`: one at each provider's annotation that saves a value outside
    /// the range of a slider in `reached`, so it starts clamped.
    pub fn warnings(&self, reached: &[Reached]) -> Vec<expr::Error<Source>> {
        let mut warnings = Vec::new();
        for slider in reached {
            let Some(slot) = self.slots.get(slider.provider) else {
                continue;
            };
            let saved = slot
                .listed
                .iter()
                .find(|entry| entry.label == slider.label)
                .and_then(|entry| entry.value);
            if let Some(saved) = saved
                && slider.spec.clamp(saved) != saved
            {
                let message = format!(
                    "{} saves slider {:?} at {}, outside its range {} to {}; it starts at {}",
                    slot.name,
                    slider.label,
                    saved,
                    slider.spec.min,
                    slider.spec.max,
                    slider.spec.clamp(saved),
                );
                warnings.push(expr::Error::eval(message, slot.listing.clone()));
            }
        }
        warnings
    }

    /// Returns an error at the annotation of the program at `played` for each
    /// entry it lists that isn't a slider it reaches and provides in
    /// `reached`, for a program that plays.
    pub fn unreached(&self, played: usize, reached: &[Reached]) -> Vec<expr::Error<Source>> {
        self.missing(played, reached, |name, label| {
            format!("{name} lists slider {label:?}, but nothing it plays reaches it")
        })
    }

    /// Returns an error at the annotation of the program at `played` for each
    /// entry it lists that isn't a slider it provides in `reached`, for a
    /// function that provides sliders to its callers.
    pub fn unprovided(&self, played: usize, reached: &[Reached]) -> Vec<expr::Error<Source>> {
        self.missing(played, reached, |name, label| {
            format!("{name} lists slider {label:?}, but provides no slider {label:?}")
        })
    }

    /// Returns an error at the annotation of the program at `played` for each
    /// entry it lists that isn't a slider it provides in `reached`, with the
    /// message `message` gives for the slot's name and the entry's label.
    fn missing(
        &self,
        played: usize,
        reached: &[Reached],
        message: impl Fn(&str, &str) -> String,
    ) -> Vec<expr::Error<Source>> {
        let Some(slot) = self.slots.get(played) else {
            return Vec::new();
        };
        slot.listed
            .iter()
            .filter(|entry| {
                !reached
                    .iter()
                    .any(|slider| slider.provider == played && slider.label == entry.label)
            })
            .map(|entry| expr::Error::eval(message(&slot.name, &entry.label), slot.listing.clone()))
            .collect()
    }
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
    /// `listed`, each entry a label or `label:saved`, in an annotation at the
    /// text's first byte.
    fn slot(index: usize, text: Range<usize>, listed: &[&str]) -> ProviderSlot {
        ProviderSlot {
            name: format!("S{index}"),
            text: text.clone(),
            listed: listed
                .iter()
                .map(|entry| match entry.split_once(':') {
                    Some((label, value)) => expr::Slider {
                        label: label.to_string(),
                        value: Some(value.parse().unwrap()),
                    },
                    None => expr::Slider {
                        label: entry.to_string(),
                        value: None,
                    },
                })
                .collect(),
            listing: Some(expr::Span::new(Source::File, text.start..text.start + 1)),
        }
    }

    /// Returns the providers of a file with slot 0 at text 10..40 and slot 1
    /// at 50..70, each listing the given labels.
    fn providers(first: &[&str], second: &[&str]) -> Providers {
        Providers {
            slots: vec![slot(0, 10..40, first), slot(1, 50..70, second)],
        }
    }

    /// Returns the id of the listing found for code written at `span` when
    /// slot 1 is played.
    fn found(providers: &Providers, source: Source, range: Range<usize>) -> Option<usize> {
        providers
            .listings(1)
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

    /// Returns the slider `label` the slot at `provider` provides, over 0 to 1.
    fn reached(provider: usize, label: &str) -> Reached {
        Reached {
            provider,
            label: label.to_string(),
            spec: slider::SliderSpec::new(0.5, 0.0, 1.0, slider::Curve::Linear).unwrap(),
        }
    }

    #[test]
    fn a_clamped_saved_value_is_a_warning() {
        let providers = providers(&["far:7"], &["near:0.5"]);
        let warnings = providers.warnings(&[reached(0, "far"), reached(1, "near")]);
        assert_eq!(
            warnings,
            [expr::Error::eval(
                "S0 saves slider \"far\" at 7, outside its range 0 to 1; it starts at 1",
                Some(expr::Span::new(Source::File, 10..11)),
            )]
        );
    }

    #[test]
    fn an_entry_the_played_program_lists_but_doesnt_reach_is_unreached() {
        let providers = providers(&["gone"], &["near", "gone"]);
        // Another slot's slider by the same label isn't the played program's.
        let reached = [reached(0, "gone"), reached(1, "near")];
        assert_eq!(
            providers.unreached(1, &reached),
            [expr::Error::eval(
                "S1 lists slider \"gone\", but nothing it plays reaches it",
                Some(expr::Span::new(Source::File, 50..51)),
            )]
        );
        assert_eq!(
            providers.unprovided(1, &reached),
            [expr::Error::eval(
                "S1 lists slider \"gone\", but provides no slider \"gone\"",
                Some(expr::Span::new(Source::File, 50..51)),
            )]
        );
    }

    #[test]
    fn an_entry_another_slot_lists_is_not_the_played_programs() {
        let providers = providers(&["unused"], &[]);
        assert_eq!(providers.unreached(1, &[]), []);
        assert_eq!(providers.unprovided(1, &[]), []);
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
