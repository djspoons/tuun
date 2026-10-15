# Where the note function runs

Labels: wayfinder:grilling
Status: open
Assignee: djspoons
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

Should evaluating the keys instrument's note function (`environment.apply_note_function`, called for every note-on in `src/lib/effects.rs`) stay on the main thread, or should note handling move off it?
