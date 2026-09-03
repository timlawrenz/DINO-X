# Ledger Reference

Every claim in this envelope maps to the **permanent** record (`docs/EXPERIMENTS_AND_RESULTS.md`)
and workstream map (`docs/EXPERIMENT_TREE.md`). The envelope is a *pointer*, not a fork —
the ledger is the living source you can trust not to have been edited to flatter this model.

---

## Findings → ledger sections

| Claim in envelope | Permanent record (EXPERIMENTS_AND_RESULTS.md) | Tree entry |
|---|---|---|
| Pan-organ failure → why specialists | § Phase 5 — `[CONCLUDED — KILL]` | `[CONCLUDED — KILL]` |
| Invalid single-organ gates (view retrieval / spacing) | § Phase 7 `[KILL]`, § Phase 8 `[PIVOT]` | — |
| Organ-specialist strategy validated | § Phase 6 (LIDC 0.728) | `[CONCLUDED — PIVOT]` |
| Hepatic-vessel + colon training, internal AUROC | § Phase 9 — `[CONCLUDED — PIVOT]` | `[CONCLUDED — PIVOT]` |
| Colon FAIL (0.6529) | § Phase 9 | `[CONCLUDED — PIVOT]` |
| Red-team leakage finding | § "External Held-Out Validation" (GO) | `[CONCLUDED — GO with caveat]` |
| **External CRLM validation (0.9413)** | § "External Held-Out Validation" — `[CONCLUDED — GO]` | `[CONCLUDED — GO with caveat]` |

## Code references (git commits on `main`)

| Artifact | Commit |
|---|---|
| Backbone + adapter, label extractor, DataLoader fix | `3880452` |
| CRLM vessel label extractor | `a556689` |
| External eval harness | `21c280f` |
| Red-team report | `672454a` |
| External validation ledger entry + tree update | `d1aab1c` |

## Reproducibility boundary
The ledger records numbers; the envelope makes them recomputable. Where a raw metric
(e.g. internal AUROC) is reported *despite* being leakage-caveated, it is flagged there
and here so a reader never mistakes it for a generalization claim.