# Local wayfinder tracker

A map and its tickets live as markdown files in this directory, standing in for issues.

- `map.md` is the map (label `wayfinder:map`).
- `tickets/NN-slug.md` is one child ticket; `NN` is its id. Its front matter holds:
  - `type`: `research`, `prototype`, `grilling` or `task` (the `wayfinder:<type>` label)
  - `status`: `open` or `closed`
  - `assignee`: who has claimed it; empty means unclaimed
  - `blocked_by`: ids of the tickets that block it
- A ticket is **unblocked** when every ticket in its `blocked_by` is `closed`. The **frontier** is the open, unblocked tickets with an empty `assignee`. To list it, run `grep -l 'status: open' tickets/*.md`, then check each result's `blocked_by` and `assignee`.
- A ticket is resolved by appending a `## Resolution` section, setting `status: closed`, and adding a line to the map's Decisions so far.
- Assets (research notes, prototypes) go in `assets/` and are linked from the ticket.
