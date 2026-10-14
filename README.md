# AgentReroute

A causal recovery framework for multi-agent workflows.

This branch implements **AgentRx-style fault attribution**: analyze a saved agent
run, check its actions, and identify the first unrecovered failure, responsible
agent, and supporting evidence.

## Quick start

Requires Python 3.10+. No third-party runtime dependencies or installation needed.

```bash
python3 -m agentrx smoke
python3 -m unittest discover -v
```

The demo runs offline with prerecorded model responses. It verifies the pipeline,
not model accuracy. Reports are saved under `runs/smoke/experiment/`.

## Benchmark data

```bash
python3 -m agentrx prepare --fetch
python3 -m agentrx audit
```

The pinned authors' release provides **73 labeled traces**: 29 retail and
44 Magentic-One. The full paper benchmark has 170; additional retail, Flash, and
RelWork data is still missing. Downloaded data and run outputs are excluded from Git.

## Run experiments

Configure a model endpoint using [the setup guide](docs/usage.md#model-backed-pilot-and-benchmark-runs), then run:

```bash
python3 -m agentrx run --config configs/pilot_gpt5.json --out runs/pilot
python3 -m agentrx run --config configs/paper_gpt5.json --out runs/gpt5
```

Runs save diagnoses, evidence, and JSON/CSV/Markdown accuracy reports. Repeat the
same command to resume; changed code, data, or model settings require a new output
directory.

**Ollama integration is planned.** Live model benchmark results have not yet been
measured. The latest verification passed 41 tests.

## Documentation

- [Usage, configuration, and implementation details](docs/usage.md)
- [Verification results and limitations](docs/verification.md)
- [Design](docs/superpowers/specs/2026-10-14-agentrx-reproduction-design.md)
- [Source attribution and licenses](THIRD_PARTY_NOTICES.md)

Independent reconstruction of [AgentRx](https://arxiv.org/abs/2602.02475), with
attributed material from [Microsoft's implementation](https://github.com/microsoft/AgentRx).
