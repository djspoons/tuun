use std::fmt::{self, Display};

use crate::{eval, expr, waveform};

// This file defines types and functions related to sliders that are platform-independent.

/// How a slider's positions map onto its range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Curve {
    /// Evenly spaced values from min to max.
    Linear,
    /// Evenly spaced ratios from min to max, as for frequencies.
    Exp,
}

impl Curve {
    /// Returns the curve named `name`, `"linear"` or `"exp"`, or `None` for
    /// any other name.
    pub fn named(name: &str) -> Option<Curve> {
        match name {
            "linear" => Some(Curve::Linear),
            "exp" => Some(Curve::Exp),
            _ => None,
        }
    }
}

impl Display for Curve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Curve::Linear => "linear",
            Curve::Exp => "exp",
        })
    }
}

/// A slider's default, range, and curve, as a `slider` call gives them.
#[derive(Clone, Debug, PartialEq)]
pub struct SliderSpec {
    pub default: f32,
    pub min: f32,
    pub max: f32,
    pub curve: Curve,
}

impl SliderSpec {
    /// Returns the spec, or a message saying why it is invalid: `min` must
    /// be less than `max`, `default` must lie in `[min, max]`, and an
    /// [`Curve::Exp`] range must be above zero.
    pub fn new(default: f32, min: f32, max: f32, curve: Curve) -> Result<SliderSpec, String> {
        // TODO an infinite min or max is accepted, which has no position and
        // isn't valid JSON for the web component.
        if min >= max {
            return Err(format!("min {} must be less than max {}", min, max));
        }
        if curve == Curve::Exp && min <= 0.0 {
            return Err(format!(
                "an \"exp\" curve needs a min above 0, but min is {}",
                min
            ));
        }
        if !(min..=max).contains(&default) {
            return Err(format!(
                "default {} is outside its range {} to {}",
                default, min, max
            ));
        }
        Ok(SliderSpec {
            default,
            min,
            max,
            curve,
        })
    }

    /// Returns the value at `position` along the slider, from 0 at `min` to 1
    /// at `max`, following the curve. Positions outside 0–1 are clamped.
    ///
    /// # Example
    ///
    /// Halfway along an exp curve is the geometric mean of the range:
    ///
    /// ```
    /// use tuun::slider::{Curve, SliderSpec};
    /// let cutoff = SliderSpec::new(1000.0, 100.0, 10000.0, Curve::Exp).unwrap();
    /// assert!((cutoff.denormalize(0.5) - 1000.0).abs() < 0.1);
    /// let mix = SliderSpec::new(0.5, 0.0, 10.0, Curve::Linear).unwrap();
    /// assert_eq!(mix.denormalize(0.5), 5.0);
    /// ```
    pub fn denormalize(&self, position: f32) -> f32 {
        let position = position.clamp(0.0, 1.0);
        match self.curve {
            Curve::Linear => self.min + position * (self.max - self.min),
            Curve::Exp => self.min * (self.max / self.min).powf(position),
        }
    }

    /// Returns the position of `value` along the slider, the inverse of
    /// [`SliderSpec::denormalize`]. Values outside the range are clamped.
    pub fn normalize(&self, value: f32) -> f32 {
        let value = value.clamp(self.min, self.max);
        match self.curve {
            Curve::Linear => (value - self.min) / (self.max - self.min),
            Curve::Exp => (value / self.min).ln() / (self.max / self.min).ln(),
        }
    }

    /// Returns `value` clamped to the range.
    pub fn clamp(&self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

impl Display for SliderSpec {
    // As in `800 (100 to 11000, exp)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({} to {}, {})",
            self.default, self.min, self.max, self.curve
        )
    }
}

/// One `slider` call: its label and spec, where it was written, and the code
/// running around it.
#[derive(Debug)]
pub struct SliderCall<S> {
    pub label: String,
    pub spec: SliderSpec,
    pub site: Option<expr::Span<S>>,
    pub path: eval::CallPath<S>,
}

impl<S> SliderCall<S> {
    /// Returns where the call was written, then where each piece of code
    /// running around it was written, innermost first.
    pub fn code(&self) -> impl Iterator<Item = &expr::Span<S>> {
        self.site.iter().chain(self.path.code())
    }

    /// Returns the offset where each call in progress around the call was
    /// written, outermost first, then the offset where the call was written.
    ///
    /// Sorting calls by it puts them in the order the text reaches them: by
    /// where the outermost call in progress was written, then the next one in,
    /// and so on. Offsets in different texts are compared as they are.
    // TODO this is a little arbitrary, but at least stable... maybe something
    // else is better?
    pub fn text_order(&self) -> Vec<usize> {
        let mut offsets: Vec<usize> = self
            .site
            .iter()
            .chain(self.path.sites())
            .map(|span| span.range.start)
            .collect();
        offsets.reverse();
        offsets
    }
}

/// Builds a `mark_id` mark around a waveform that ramps linearly from
/// `last_value` to `new_value` over `ramp_duration_secs`, then holds
/// `new_value` forever.
// TODO: consider an exponential ramp
pub fn make_ramp<M>(
    mark_id: M,
    last_value: f32,
    new_value: f32,
    ramp_duration_secs: f32,
) -> waveform::Waveform<M> {
    use waveform::BinaryOperator;
    use waveform::Waveform::{Append, BinaryOp, Const, Fin, Marked, Time};

    let ramp = Append(
        Box::new(Fin {
            length: Box::new(BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Time(())),
                Box::new(Const(ramp_duration_secs)),
            )),
            waveform: Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(Time(())),
                    Box::new(Const((new_value - last_value) / ramp_duration_secs)),
                )),
                Box::new(Const(last_value)),
            )),
            state: (),
        }),
        Box::new(Const(new_value)),
        (),
    );
    Marked {
        id: mark_id,
        waveform: Box::new(ramp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_needs_min_below_max() {
        let error = SliderSpec::new(1.0, 1.0, 1.0, Curve::Linear).unwrap_err();
        assert_eq!(error, "min 1 must be less than max 1");
        assert!(SliderSpec::new(1.0, 2.0, 1.0, Curve::Linear).is_err());
    }

    #[test]
    fn an_exp_spec_needs_a_range_above_zero() {
        let error = SliderSpec::new(1.0, 0.0, 10.0, Curve::Exp).unwrap_err();
        assert_eq!(error, "an \"exp\" curve needs a min above 0, but min is 0");
        // The same range is fine on a linear curve.
        assert!(SliderSpec::new(1.0, 0.0, 10.0, Curve::Linear).is_ok());
    }

    #[test]
    fn a_spec_default_lies_in_its_closed_range() {
        assert!(SliderSpec::new(0.1, 0.1, 1.1, Curve::Linear).is_ok());
        assert!(SliderSpec::new(1.1, 0.1, 1.1, Curve::Linear).is_ok());
        let error = SliderSpec::new(2.0, 0.1, 1.1, Curve::Linear).unwrap_err();
        assert_eq!(error, "default 2 is outside its range 0.1 to 1.1");
    }

    #[test]
    fn positions_and_values_are_inverses_on_both_curves() {
        for curve in [Curve::Linear, Curve::Exp] {
            let spec = SliderSpec::new(800.0, 100.0, 11000.0, curve).unwrap();
            assert_eq!(spec.denormalize(0.0), 100.0);
            assert!((spec.denormalize(1.0) - 11000.0).abs() < 0.01);
            for value in [100.0, 800.0, 5000.0] {
                assert!((spec.denormalize(spec.normalize(value)) - value).abs() < 0.01);
            }
        }
    }

    #[test]
    fn positions_and_values_outside_the_slider_are_clamped() {
        let spec = SliderSpec::new(0.5, 0.0, 1.0, Curve::Linear).unwrap();
        assert_eq!(spec.denormalize(1.5), 1.0);
        assert_eq!(spec.normalize(-3.0), 0.0);
        assert_eq!(spec.clamp(2.0), 1.0);
    }

    #[test]
    fn calls_are_in_text_order_by_their_outermost_call_first() {
        let call = |site: usize, path: &[usize]| SliderCall {
            label: "mix".to_string(),
            spec: SliderSpec::new(0.5, 0.0, 1.0, Curve::Linear).unwrap(),
            site: Some(expr::Span::new((), site..site + 1)),
            path: path.iter().fold(eval::CallPath::empty(), |path, &at| {
                path.enter(None, Some(expr::Span::new((), at..at + 1)))
            }),
        };
        // Written at 30, inside a call at 5 inside a call at 20.
        let inner = call(30, &[20, 5]);
        assert_eq!(inner.text_order(), [20, 5, 30]);
        // A call written at 10, outside any call, comes before it.
        assert!(call(10, &[]).text_order() < inner.text_order());
        assert!(call(2, &[20]).text_order() < inner.text_order());
    }

    #[test]
    fn a_call_is_inside_its_running_code() {
        let call = SliderCall {
            label: "mix".to_string(),
            spec: SliderSpec::new(0.5, 0.0, 1.0, Curve::Linear).unwrap(),
            site: Some(expr::Span::new((), 3..8)),
            path: eval::CallPath::empty()
                .enter(Some(expr::Span::new((), 0..20)), None)
                .enter(
                    Some(expr::Span::new((), 30..40)),
                    Some(expr::Span::new((), 2..9)),
                ),
        };
        let code: Vec<_> = call.code().map(|span| span.range.clone()).collect();
        assert_eq!(code, [3..8, 30..40, 0..20]);
        // Where calls were written orders the call, not where code was.
        assert_eq!(call.text_order(), [2, 3]);
    }
}
