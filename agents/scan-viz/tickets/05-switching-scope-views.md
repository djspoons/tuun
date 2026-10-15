# Switching scope views

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

How does the performer switch between Snapshot and Scan: which computer-keyboard key (and in which editor modes), which Launchkey button if any, which scope view is the default, and does the choice persist across runs?

## Resolution

- **Key: Ctrl+S** toggles between Snapshot and Scan. It is unbound today.
- **Editor modes:** accepted in every editor mode where possible, including Keys and Record. In Keys mode it is caught before the piano-key mapping, as Ctrl+C is (S is a piano key there).
- **Launchkey:** no button for now.
- **Default:** Snapshot, so the screen is unchanged for anyone who never presses Ctrl+S.
- **Persistence:** none across runs; a CLI flag (for example `--scope-view scan`, in the kebab-case most flags get from clap, like `--sample-rate`; only `--beats_per_measure` is spelled with an underscore) chooses the scope view at startup.
- **Feedback:** a status message on each switch, for example "Scope view: Scan", like the other app-wide toggles.

