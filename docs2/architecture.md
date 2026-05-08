# SCOPE Architecture

SCOPE uses a fixed main workflow with conditional capability triggers. The intended controller is Codex.

```text
prompt
-> decompose
-> conditional augment
   -> retriever when external reference grounding is required
   -> reasoner when semantic reasoning is required
-> synthesize
-> generate
-> verify
-> conditional repair
   -> repairer when verification fails or is uncertain
-> artifacts and metrics
```

## Agent-Facing Layer

The `.codex/skills` directory exposes SCOPE as an agent-facing workflow. It is the source of truth for when and how Codex should call route-changing workflow skills.

The canonical workflow spec is split across:

- `SCOPE/.codex/skills/scope-agentic-generation/SKILL.md`
- `SCOPE/.codex/skills/scope-agentic-generation/references/main-flow.md`
- `SCOPE/.codex/skills/scope-agentic-generation/references/stage-contracts.md`

The main skill owns the orchestration rule:

```text
decompose
-> if external_reference unknowns exist: retrieve
-> if semantic_reasoning unknowns exist: reason
-> synthesize
-> generate
-> verify
-> if verification introduces new unknowns: retrieve/reason, then synthesize
-> else if verification fails or is uncertain: repair
-> generate/verify again
-> finalize
-> eval
```

Only `retrieve`, `reason`, and `repair` are route-changing workflow branch skills. Evaluation and benchmark utilities are exposed through the Python CLI commands documented in the README.

## Runtime Layer

The Python package under `src/scope` owns the stage tools and stable artifact contract. It does not need to own the top-level orchestration during normal agent-driven runs.

Its runtime configuration is intentionally split so we do not slide back into a CARE-style internal controller:

- `controller`: always the outer agent in the default path.
- `image_gen`: CARE-style provider/base_url/model wiring for actual image generation.
- `judge`: separate provider/base_url/model wiring for verification.

For `decompose`, `retrieve`, `reason`, `synthesize`, and `repair`, the normal path is agent-first:

- Codex writes an input artifact such as `decomposition.input.json` or `repair.input.json`
- the corresponding stage validates the payload and persists the canonical artifact
- downstream stages read the persisted canonical artifact, not chat history

For reasoning specifically, the canonical artifact keeps a light `unknown_id + note` shape. The note is the resolved semantic result, not a trace of the thinking process. Synthesis reads that result together with the original constraints, replaces unresolved semantic phrases with the resolved value, and writes the final prompt. It should not keep source expressions or clues unless the user asked to show them.

In the normal Codex path, those stage input artifacts are required. Runtime fallbacks are kept only for internal smoke or debug code paths, not for the public staged workflow.

The stage CLI is therefore a tool layer:

- it validates payloads
- it persists canonical artifacts
- it performs deterministic backend calls

It is not the canonical place to interpret workflow routing in the normal Codex path.

## Evaluation Layer

Runs should write standard files:

- `state.json`
- `final_prompt.txt`
- generated image files such as `iteration_XX.image.png`
- `metrics.json`

The stage-specific canonical artifacts are also part of the contract:

- `decomposition.json`
- `reasoning.json`
- `synthesis.json`
- `verification.json`
- `repair_decision.json`

Metrics should be computed from artifacts, not from chat transcripts.
