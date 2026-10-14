# Implementation verification — 2026-10-14

The reconstruction is implemented. This records software verification, not
reproduction of the paper's measured model accuracies.

## Evidence

- `python3 -m unittest discover -v`: 41 passing tests, including CLI subprocesses,
  real localhost HTTP transport/retries, Python worker processes, prefix isolation,
  attribution/abstention validation, fixed denominators, and resume identities.
- `python3 -m compileall -q agentrx` and `git diff --check`: passed.
- Editable installation and standalone wheel build: passed. The wheel was
  installed into a separate target and its smoke test ran from `/tmp`, confirming
  packaged resources and worker paths work outside the checkout.
- `prepare --fetch`: fetched the pinned authors' commit and audited 73 pairs with
  no excluded records. The original corpus first-prefix prompt boundary audit
  also passed for all 73 cases.
- Gold-label oracle consistency check: all 73 original roots can be accepted and
  scored correctly, including two whose responsible agent differs from the
  selected event speaker. This is a scorer check by construction, not prediction
  accuracy.
- `smoke --out runs/verified-smoke`: baseline and AgentRx plumbing across three
  repetitions each; six jobs finished without errors. Reports visibly identify
  offline fixture testing. Standalone replay diagnosis found two count-mismatch
  violations and attributed the fixture failure to step 3.
- Model preflight: correctly refused to start without `OPENAI_API_KEY`, exited 2,
  and did not create a model-backed experiment. No paid model calls were made.

## Independent review and fixes

A fresh reviewer inspected the complete implementation and ran the original 36
tests. Four material findings were verified with failing regression tests, then
fixed; the full 41-test suite passed afterward:

1. Raw module exports exposed file access through `statistics.sys`; constrained
   export facades now reject this and dynamic attribute/frame variants.
2. Responsible-agent validation incorrectly required the agent to speak at the
   selected step; validation now checks known agents across the trajectory.
3. Invalid list/range selectors silently matched or skipped; malformed selectors
   now become explicit errors, excluding booleans and negative/reversed ranges.
4. Localization abstention required a category while its prompt deferred category;
   null localization with a rationale/custom category now becomes Inconclusive.

No minor findings were deferred. The reviewer could not assess live endpoint
compatibility/diagnostic quality, exact numeric replication/full-corpus membership,
or unavailable historical prompt/model snapshots. Those remain unverified and
are disclosed as external prerequisites, not treated as passing checks.

## Implementation decisions and limits

- Execution continued inline following the user's approval, in the initially
  empty project on a dedicated branch. There was no existing base branch or
  remote to merge or publish; all project work remains local and reviewable.
- Actual domain-registry tool definitions replace the source's misleading
  example-event `tools_structure.json`; MIT notices and revision are retained.
- DeepSeek requires a configured endpoint serving V3.2; a changing API alias is
  not silently substituted. Incorrect endpoint configuration can still change
  experimental meaning, so returned model identity is recorded.
- Python execution exposes a deliberately narrower interface than unrestricted
  reference execution. Unsupported generated constructs become auditable errors;
  this may reduce checker coverage. It is not a complete adversarial sandbox.
- Paper sampling/deployment snapshots are unspecified; completion caps, prompt
  reconstruction, and sample SD are explicit reconstruction choices.

The available original data consists of 29 retail and 44 Magentic-One traces.
Full 170-trajectory reproduction still lacks 10 retail, 42 Flash, and 45 RelWork
traces, working model credentials, and completed independent live repetitions.
W&W predictions and normalized MAST data are import interfaces, not bundled
reproductions of those external systems.
