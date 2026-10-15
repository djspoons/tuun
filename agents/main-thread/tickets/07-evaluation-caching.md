# Caching evaluated modules and bindings

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Should processing cache what it now evaluates again for every program, so results arrive sooner? Three costs dominate:
- the exports of each opened or used module (and its dependencies);
- the named bindings earlier in the file, which makes a reload quadratic in the number of named programs;
- the second evaluation of `on_beats` programs in `evaluated_waveform`.

Measured in [Where program processing runs](06-evaluation.md): evaluation is about 93–95% of a reload, and nearly all of the time to leave Edit mode. This shortens the "Processing…" wait, not input latency.

## Out of scope

Closed without a decision. Program processing runs on the processing thread ([Where program processing runs](06-evaluation.md)), so caching there shortens the "Processing…" wait but not input latency, and this map ends at a spec about input latency. Caching bindings for the main thread's Edit-mode helpers stays with the cost of those helpers in the map's Not yet specified. Caching for the processing thread can come back as a separate effort about the "Processing…" wait.
