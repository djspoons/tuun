//! User-visible status messages: the one the status line shows, and every
//! posted message, queued for echoing to the terminal.

use std::time::{Duration, Instant};

use crate::diagnostics::{self, Diagnostic};

/// How long an `Info` message stays on the status line.
pub const INFO_LIFETIME: Duration = Duration::from_secs(5);

/// The importance of a message, from least to most severe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// A confirmation of what just happened; fades after [`INFO_LIFETIME`].
    Info,
    /// Something the user should act on; stays until replaced or cleared.
    Warning,
    /// Something failed; stays until replaced or cleared.
    Error,
}

/// A posted message.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub severity: Severity,
    /// The message text. May be multi-line (e.g. an error followed by its
    /// source snippet): the first line is the summary, and single-line
    /// display sites should show only that line.
    pub text: String,
    pub posted_at: Instant,
}

/// The status line's current message and the messages not yet echoed.
///
/// Posts are grouped into batches, one per user event. Within a batch, a
/// post doesn't replace a more severe one, so an error raised while
/// handling an event isn't hidden by a confirmation posted later for the
/// same event. A post in a later batch always replaces the current message.
///
/// # Example
/// ```
/// use tuun::messages::{Messages, Severity};
///
/// let mut messages = Messages::default();
/// messages.begin_batch();
/// messages.post(Severity::Error, "Error: 1:1: unbound variable 'x'");
/// messages.post(Severity::Info, "Playing waveform A:1");
/// assert_eq!(messages.current().unwrap().severity, Severity::Error);
///
/// messages.begin_batch();
/// messages.post(Severity::Info, "Stopped program A:1");
/// assert_eq!(messages.current().unwrap().text, "Stopped program A:1");
/// assert_eq!(messages.take_unechoed().len(), 3);
/// ```
#[derive(Debug, Default)]
pub struct Messages {
    current: Option<Message>,
    /// The batch `current` was posted in.
    current_batch: u64,
    batch: u64,
    unechoed: Vec<Message>,
}

impl Messages {
    /// Starts a new batch of posts.
    pub fn begin_batch(&mut self) {
        self.batch += 1;
    }

    /// Posts `text` with the given severity, making it the current message
    /// unless the current one is more severe and was posted in this batch.
    pub fn post(&mut self, severity: Severity, text: impl Into<String>) {
        let message = Message {
            severity,
            text: text.into(),
            posted_at: Instant::now(),
        };
        self.unechoed.push(message.clone());
        let outranked = self
            .current
            .as_ref()
            .is_some_and(|current| self.current_batch == self.batch && current.severity > severity);
        if !outranked {
            self.current = Some(message);
            self.current_batch = self.batch;
        }
    }

    /// Posts `diagnostics` as one message (see
    /// [`diagnostics::error_message`]): an error if any of them is an error,
    /// otherwise a warning. Posts nothing when `diagnostics` is empty.
    pub fn post_diagnostics(&mut self, diagnostics: &[Diagnostic]) {
        if diagnostics.is_empty() {
            return;
        }
        let severity = if diagnostics
            .iter()
            .any(|d| d.severity == diagnostics::Severity::Error)
        {
            Severity::Error
        } else {
            Severity::Warning
        };
        self.post(severity, diagnostics::error_message(diagnostics));
    }

    /// Clears the current message.
    pub fn clear(&mut self) {
        self.current = None;
    }

    /// Returns the current message, regardless of its age.
    pub fn current(&self) -> Option<&Message> {
        self.current.as_ref()
    }

    /// Returns the current message if it should still be shown at `now`:
    /// an `Info` message only within [`INFO_LIFETIME`] of being posted.
    pub fn visible(&self, now: Instant) -> Option<&Message> {
        self.current.as_ref().filter(|message| {
            message.severity > Severity::Info
                || now.saturating_duration_since(message.posted_at) < INFO_LIFETIME
        })
    }

    /// Returns every message posted since the last call, oldest first.
    pub fn take_unechoed(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.unechoed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn less_severe_post_in_same_batch_keeps_current() {
        let mut messages = Messages::default();
        messages.begin_batch();
        messages.post(Severity::Warning, "File changed on disk");
        messages.post(Severity::Info, "Playing");
        assert_eq!(messages.current().unwrap().text, "File changed on disk");
    }

    #[test]
    fn equal_or_more_severe_post_in_same_batch_replaces_current() {
        let mut messages = Messages::default();
        messages.begin_batch();
        messages.post(Severity::Info, "Recorded 1 note");
        messages.post(Severity::Info, "Playing");
        assert_eq!(messages.current().unwrap().text, "Playing");
        messages.post(Severity::Error, "Error: bad");
        assert_eq!(messages.current().unwrap().text, "Error: bad");
    }

    #[test]
    fn post_in_later_batch_replaces_more_severe_current() {
        let mut messages = Messages::default();
        messages.begin_batch();
        messages.post(Severity::Error, "Error: bad");
        messages.begin_batch();
        messages.post(Severity::Info, "Playing");
        assert_eq!(messages.current().unwrap().text, "Playing");
    }

    #[test]
    fn clear_removes_current_but_not_unechoed() {
        let mut messages = Messages::default();
        messages.post(Severity::Error, "Error: bad");
        messages.clear();
        assert!(messages.current().is_none());
        assert_eq!(messages.take_unechoed().len(), 1);
        assert!(messages.take_unechoed().is_empty());
    }

    #[test]
    fn diagnostics_post_as_their_most_severe() {
        let mut messages = Messages::default();
        let warning = Diagnostic::message_only("unused".to_string()).as_warning();
        messages.post_diagnostics(std::slice::from_ref(&warning));
        assert_eq!(messages.current().unwrap().severity, Severity::Warning);
        messages.post_diagnostics(&[warning, Diagnostic::message_only("bad".to_string())]);
        assert_eq!(messages.current().unwrap().severity, Severity::Error);
    }

    #[test]
    fn only_info_expires() {
        let mut messages = Messages::default();
        messages.post(Severity::Info, "Playing");
        let posted_at = messages.current().unwrap().posted_at;
        assert!(messages.visible(posted_at).is_some());
        assert!(messages.visible(posted_at + INFO_LIFETIME).is_none());

        messages.post(Severity::Warning, "File changed on disk");
        let posted_at = messages.current().unwrap().posted_at;
        assert!(messages.visible(posted_at + INFO_LIFETIME * 10).is_some());
    }
}
