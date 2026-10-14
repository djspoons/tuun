# Wayfinding tracker: local Markdown

This directory holds one wayfinder map and its tickets as Markdown files, standing in for issues.

## Wayfinding operations

- **Map**: `map.md`, labelled `wayfinder:map`.
- **Ticket**: `tickets/NN-slug.md`, a child of the map. `NN` is its id; its `# ` heading is its name.
- **Header fields** at the top of each ticket:
  - `Labels:` one `wayfinder:<type>` label (`research`, `prototype`, `grilling`, `task`).
  - `Status:` `open` or `closed`.
  - `Assignee:` empty when unclaimed. Claim a ticket by setting it, and commit and push, before any other work.
  - `Blocked by:` links to the tickets that block it, or `none`. This is the blocking edge.
- **Frontier**: tickets with `Status: open`, an empty `Assignee:`, and every `Blocked by:` ticket closed.
- **Resolve**: add a `## Resolution` section to the ticket, set `Status: closed`, and add a line to the map's Decisions so far.
- **Assets** (research findings, prototypes) live under `research/` or `prototypes/` and are linked from the ticket.
