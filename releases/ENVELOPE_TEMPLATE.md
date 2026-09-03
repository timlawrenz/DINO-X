# DINO-X Model Release Envelope

Every released model ships with an **evidence envelope** — a self-contained bundle
that lets anyone follow the science from ground truth, not just load a binary.
A release is not a checkpoint + a model card; it is the entire decision trail that
made the checkpoint trustworthy. The envelope is the open-source thought in action:
we publish our reasoning, our failures, and our verification, not just our results.

---

## Directory Layout

```
{model-slug}/
├── ENVELOPE.md              # THIS file — entry point, how to follow the science
├── model_card.md            # HF model card (also published to HF Hub)
├── evaluation.json          # All gate results, machine-readable
├── provenance.yaml          # Git SHAs, data snapshot hashes, frozen configs
├── source_branch            # Git branch/tag this model was built from
├── the_science/             # Prose walkthrough — decision by decision
│   ├── 01_decision_trail.md # Why this model exists; pivots, dead-ends, failures
│   ├── 02_training_recipe.md# Exact recipe: data, preprocessing, hyperparameters
│   ├── 03_validation.md     # Every test we ran and how we know it's right
│   ├── 04_negative_results.md # What we tried that did NOT work, and the evidence
│   └── 05_known_limits.md   # Where the model still fails, honestly
├── reproduce/               # Reproduce every headline number from scratch
│   ├── README.md            # The exact commands, in order
│   └── repro_metrics.py     # Script that loads the model and recomputes the gates
└── experiments/             # Pointers to ledger/tree entries (the permanent record)
    ├── LEDGER_REFERENCE.md  # Exact file + section for every claim
    └── artifacts/           # (small) copies or pointers to key evidence
```

---

## The Non-Negotiable Principles

1. **Every headline number is traceable to an artifact.** No number appears in the
   envelope unless `reproduce/repro_metrics.py` (or a pointer to the ledger) can
   recompute it from a named file. See the adversarial pass in
   `scientific-experiment-structure` skill.

2. **Failures are published with the same rigor as successes.** Each `04_negative_results.md`
   records what was tried, the number that falsified it, and the git commit. Negative
   results are credibility assets, not embarrassment.

3. **The source branch is pinned.** `source_branch` names the exact git ref. Combined
   with `provenance.yaml` commit SHAs, the model is reproducible as-of-a-moment, not
   "latest."

4. **Reproduction is a first-class deliverable.** `reproduce/repro_metrics.py` runs on
   CPU or the stated GPU and recomputes the gates. If a number can't be reproduced
   mechanically, it is not a release.

5. **Every claim maps to the permanent ledger.** `experiments/LEDGER_REFERENCE.md`
   cites the exact `docs/EXPERIMENTS_AND_RESULTS.md` section and
   `docs/EXPERIMENT_TREE.md` entry for each finding, so the envelope is a *pointer*
   into the living record, never a fork that can drift.

---

## How To Follow The Science (reading order)

For a newcomer who wants to trust this model:

1. Read `ENVELOPE.md` (this file) — the map.
2. Read `the_science/01_decision_trail.md` — *why* this model exists, including the
   failures that shaped it.
3. Read `the_science/02_training_recipe.md` — *exactly* what went in.
4. Read `the_science/03_validation.md` — *how we know the numbers are real*, including
   adversarial passes and any external held-out validation.
5. Read `the_science/04_negative_results.md` — *what we tried that failed*, and the
   evidence.
6. Read `the_science/05_known_limits.md` — *where it still breaks*.
7. Run `reproduce/repro_metrics.py` — reproduce the headline numbers yourself.

---

## Generating The Envelope

A helper script scaffolds a new envelope and validates it. It **auto-populates**
`02_training_recipe.md` and `provenance.yaml` from the frozen `finetune_config.json`
and backbone `config.json` — so the numbers come from disk, never memory. It does NOT
invent the honest narrative (decision trail / validation / negative results / limits are
written by a human who ran the work).

```bash
# Build a new envelope from an adapter dir
python scripts/release/build_envelope.py build \
  --adapter-path adapters/{task}-{backbone}-{variant} \
  --model-slug dinox-ct-{organ}-{backbone}-v1 \
  --internal-auc {auc} [--leakage-caveat "..."]

# Validate an existing envelope (all required files present)
python scripts/release/build_envelope.py validate --model-slug {slug}
```

Validation rules the generator enforces:
- `ENVELOPE.md`, `model_card.md`, `evaluation.json`, `provenance.yaml`,
  `source_branch`, and all 5 `the_science/` files + `reproduce/README.md` exist.
- Recipe + provenance auto-populated from frozen configs (not hand-typed).
- Every key in `evaluation.json` has a corresponding claim in `03_validation.md`.
- `reproduce/repro_metrics.py` exists for mechanical metric reproduction.

The human still writes: `01_decision_trail.md`, `03_validation.md`,
`04_negative_results.md`, `05_known_limits.md`, and the model-card narrative. That is
the point — the envelope's credibility comes from the honest story being told, which
cannot be faked by a scaffolder.