# The 10 ms status wait

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

Should the main loop stop blocking for up to 10 ms (`recv_timeout`) while it waits for a tracker status, and if so, what paces the loop instead? The Scope views map (`agents/scan-viz/` on `claude/scan-viz`) changes how samples reach the UI but leaves this status drain unchanged.
