# Instrument the main loop

Labels: wayfinder:task
Status: open
Assignee:
Blocked by: none

## Question

Add timing to the native UI so later decisions have numbers: how long each main-loop step takes (SDL events, Launchkey/MIDI events, the status wait, the recording tick, render, the Launchkey update) and how long each input event takes from arrival to its command being sent to the tracker, for the computer keyboard, Launchkey pads and Launchkey piano keys.

Decided while charting: keep the timings in the existing `Metric` (`src/lib/metric.rs`) and show them on screen like tracker load, behind a CLI flag. Done when the instrumented build compiles and passes the `CLAUDE.md` checks, and the flag and the numbers it shows are documented here.
