use std::time::Duration;

use crate::metric::Metric;

/// One step of a main-loop pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    SdlEvents,
    LaunchkeyEvents,
    StatusWait,
    Tick,
    Render,
    LaunchkeyUpdate,
}

impl Step {
    const ALL: [Step; 6] = [
        Step::SdlEvents,
        Step::LaunchkeyEvents,
        Step::StatusWait,
        Step::Tick,
        Step::Render,
        Step::LaunchkeyUpdate,
    ];

    fn label(self) -> &'static str {
        match self {
            Step::SdlEvents => "SDL events",
            Step::LaunchkeyEvents => "Launchkey events",
            Step::StatusWait => "status wait",
            Step::Tick => "tick",
            Step::Render => "render",
            Step::LaunchkeyUpdate => "Launchkey update",
        }
    }
}

/// A source of input events whose latency is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    Keyboard,
    Pad,
    PianoKey,
}

impl Input {
    const ALL: [Input; 3] = [Input::Keyboard, Input::Pad, Input::PianoKey];

    fn label(self) -> &'static str {
        match self {
            Input::Keyboard => "keyboard input",
            Input::Pad => "pad input",
            Input::PianoKey => "piano key input",
        }
    }
}

/// The worst recent times of each main-loop step and each input's latency.
///
/// Times are kept in milliseconds, as the largest value seen in each second of
/// the last ten seconds.
pub struct LoopTiming {
    steps: Vec<Metric<f32>>,
    inputs: Vec<Metric<f32>>,
}

impl Default for LoopTiming {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopTiming {
    /// Returns timings with nothing recorded.
    pub fn new() -> Self {
        let metric = || Metric::new(Duration::from_secs(10), 10);
        LoopTiming {
            steps: Step::ALL.iter().map(|_| metric()).collect(),
            inputs: Input::ALL.iter().map(|_| metric()).collect(),
        }
    }

    /// Records how long one run of `step` took.
    pub fn record_step(&mut self, step: Step, elapsed: Duration) {
        self.steps[step as usize].set_max(millis(elapsed));
    }

    /// Records the time from an `input` event's arrival to the end of its handling.
    pub fn record_input(&mut self, input: Input, latency: Duration) {
        self.inputs[input as usize].set_max(millis(latency));
    }

    /// Returns one line per step and input: its worst time in the current second
    /// and its worst time over the last ten seconds.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use tuun::loop_timing::{LoopTiming, Step};
    ///
    /// let mut timing = LoopTiming::new();
    /// timing.record_step(Step::Render, Duration::from_micros(4200));
    /// assert_eq!(timing.lines()[4], "render: 4.2 ms (10 s max 4.2)");
    /// ```
    pub fn lines(&mut self) -> Vec<String> {
        let steps = Step::ALL.iter().map(|s| s.label());
        let inputs = Input::ALL.iter().map(|i| i.label());
        steps
            .chain(inputs)
            .zip(self.steps.iter_mut().chain(self.inputs.iter_mut()))
            .map(|(label, metric)| {
                let values: Vec<f32> = metric.iter().collect();
                let current = values.last().copied().unwrap_or_default();
                let max = values.iter().copied().fold(0.0, f32::max);
                format!("{label}: {current:.1} ms (10 s max {max:.1})")
            })
            .collect()
    }
}

fn millis(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use mock_instant::thread_local::MockClock;

    #[test]
    fn lines_show_the_current_second_and_the_ten_second_max() {
        MockClock::set_time(Duration::from_secs(60));
        let mut timing = LoopTiming::new();
        timing.record_input(Input::Pad, Duration::from_millis(12));
        MockClock::advance(Duration::from_secs(1));
        timing.record_input(Input::Pad, Duration::from_millis(3));

        let lines = timing.lines();
        assert_eq!(lines.len(), 9);
        assert_eq!(lines[0], "SDL events: 0.0 ms (10 s max 0.0)");
        assert_eq!(lines[7], "pad input: 3.0 ms (10 s max 12.0)");
    }
}
