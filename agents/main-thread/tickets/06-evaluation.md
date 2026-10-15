# Where program evaluation runs

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: none

## Question

Should evaluating source (parsing, type checking and evaluating programs) move off the main thread, and if so, how do its results get back to the main loop? Measured in [Measure on the dev's machine](02-measure.md): reloading a large source file blocks the main thread for about 2.6 s, and leaving Edit mode with Return (which evaluates the program) for about 233 ms. Every input arriving meanwhile waits.
