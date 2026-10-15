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

/// One part of the render step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPart {
    /// Loading fonts from disk.
    Fonts,
    /// Rendering text to surfaces and creating textures from them.
    Text,
    /// Drawing everything else, before `present()`.
    Other,
    /// `present()`, including any wait for vsync.
    Present,
}

impl RenderPart {
    const ALL: [RenderPart; 4] = [
        RenderPart::Fonts,
        RenderPart::Text,
        RenderPart::Other,
        RenderPart::Present,
    ];

    fn label(self) -> &'static str {
        match self {
            RenderPart::Fonts => "render fonts",
            RenderPart::Text => "render text",
            RenderPart::Other => "render other",
            RenderPart::Present => "render present",
        }
    }
}

/// One Edit-mode helper that runs while handling a keystroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Helper {
    /// Completing the name before the cursor, or inserting a parameter hint
    /// after a `(`.
    Complete,
    /// Showing the type of the name under the cursor.
    ShowType,
    /// Inserting or deleting text, which re-parses the program.
    TextEdit,
}

impl Helper {
    const ALL: [Helper; 3] = [Helper::Complete, Helper::ShowType, Helper::TextEdit];

    fn label(self) -> &'static str {
        match self {
            Helper::Complete => "complete",
            Helper::ShowType => "show type",
            Helper::TextEdit => "text edit",
        }
    }
}

/// The worst recent times of each main-loop step, each input's latency, each
/// part of the render step and each Edit-mode helper, and the most text
/// textures created in one frame.
///
/// Values are kept as the largest seen in each second of the last ten seconds;
/// times are in milliseconds.
pub struct LoopTiming {
    steps: Vec<Metric<f32>>,
    inputs: Vec<Metric<f32>>,
    render_parts: Vec<Metric<f32>>,
    helpers: Vec<Metric<f32>>,
    text_textures: Metric<f32>,
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
            render_parts: RenderPart::ALL.iter().map(|_| metric()).collect(),
            helpers: Helper::ALL.iter().map(|_| metric()).collect(),
            text_textures: metric(),
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

    /// Records how long `part` of one render took.
    pub fn record_render_part(&mut self, part: RenderPart, elapsed: Duration) {
        self.render_parts[part as usize].set_max(millis(elapsed));
    }

    /// Records how long one run of `helper` took.
    pub fn record_helper(&mut self, helper: Helper, elapsed: Duration) {
        self.helpers[helper as usize].set_max(millis(elapsed));
    }

    /// Records how many text textures one render created.
    pub fn record_text_textures(&mut self, count: u32) {
        self.text_textures.set_max(count as f32);
    }

    /// Returns one line per step, input, render part and helper, with its worst time in
    /// the current second and its worst time over the last ten seconds, then a
    /// line with the most text textures created in one frame, likewise.
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
        let parts = RenderPart::ALL.iter().map(|p| p.label());
        let helpers = Helper::ALL.iter().map(|h| h.label());
        let mut lines: Vec<String> = steps
            .chain(inputs)
            .chain(parts)
            .chain(helpers)
            .zip(
                self.steps
                    .iter_mut()
                    .chain(self.inputs.iter_mut())
                    .chain(self.render_parts.iter_mut())
                    .chain(self.helpers.iter_mut()),
            )
            .map(|(label, metric)| {
                let (current, max) = current_and_max(metric);
                format!("{label}: {current:.1} ms (10 s max {max:.1})")
            })
            .collect();
        let (current, max) = current_and_max(&mut self.text_textures);
        lines.push(format!("text textures: {current} (10 s max {max})"));
        lines
    }
}

fn current_and_max(metric: &mut Metric<f32>) -> (f32, f32) {
    let values: Vec<f32> = metric.iter().collect();
    let current = values.last().copied().unwrap_or_default();
    let max = values.iter().copied().fold(0.0, f32::max);
    (current, max)
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
        assert_eq!(lines.len(), 17);
        assert_eq!(lines[0], "SDL events: 0.0 ms (10 s max 0.0)");
        assert_eq!(lines[7], "pad input: 3.0 ms (10 s max 12.0)");
    }

    #[test]
    fn lines_end_with_render_parts_helpers_and_text_textures() {
        MockClock::set_time(Duration::from_secs(60));
        let mut timing = LoopTiming::new();
        timing.record_render_part(RenderPart::Present, Duration::from_micros(1500));
        timing.record_helper(Helper::TextEdit, Duration::from_micros(800));
        timing.record_text_textures(40);
        timing.record_text_textures(25);

        let lines = timing.lines();
        assert_eq!(lines[12], "render present: 1.5 ms (10 s max 1.5)");
        assert_eq!(lines[15], "text edit: 0.8 ms (10 s max 0.8)");
        assert_eq!(lines[16], "text textures: 40 (10 s max 40)");
    }
}
