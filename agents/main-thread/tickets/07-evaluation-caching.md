# Caching evaluated modules and bindings

Labels: wayfinder:grilling
Status: open
Assignee: djspoons
Blocked by: none

## Question

Should processing cache what it now evaluates again for every program, so results arrive sooner? Three costs dominate:
- the exports of each opened or used module (and its dependencies);
- the named bindings earlier in the file, which makes a reload quadratic in the number of named programs;
- the second evaluation of `on_beats` programs in `evaluated_waveform`.

Measured in [Where program processing runs](06-evaluation.md): evaluation is about 93–95% of a reload, and nearly all of the time to leave Edit mode. This shortens the "Processing…" wait, not input latency.
