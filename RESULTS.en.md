[中文](RESULTS.md) | English

# Table of all variants

One row per tested variant. All numbers are **changes relative to that row's baseline**; rows with different baselines are not directly comparable (baseline definitions in [reports/00](reports/00-methodology.md) §2; the reports are in Chinese):

| Baseline | Meaning |
|---|---|
| A | Production config on 2026-09-24: vLLM 0.28, `-fast` (int4 LM head), MTP k=4, full-vocabulary drafting, KV bf16 |
| P | Production config on 2026-09-25: A with the int8 LM head, measured on a test card |
| off | Same config as P, with the hybrid switch off, on another node |
| H8 | int8-LM-head production replica serving live traffic (same-window A/B) |
| shuffle | litellm simple-shuffle on the same pair of replicas |

Columns:
- **Single**: c1 per-stream decode tok/s median, EN / ZH.
- **c8**: 8-concurrent aggregate tok/s, EN / ZH.
- **long32k**: decode tok/s with a 33K-token input.
- **Dev-agent**: devbench aggregate tok/s, c1 / c8 (in parentheses: c1 per-turn decode median). Noise band: c1 ±0.5%, c8 ±5%.
- **Precision**: gate = number of the 17 greedy outputs that are identical token by token; tf = teacher-forcing top-1 agreement, noise floor 99.51% (P round) / 99.45% (A round).
- General-load noise band ±2%. "—" means not measured.

## Shipped (3 directions)

| Variant | Baseline | Single EN / ZH | c8 EN / ZH | long32k | Dev-agent c1 / c8 | Precision | Verdict | Report |
|---|---|---|---|---|---|---|---|---|
| **int8 LM head** (D2) | A | −4.2% / −0.1% | −2.0% / −4.4% | +1.3% (long4k −7.2%) | −1.5% / −1.4% (decode −0.2%; c8 within noise) | gate 0/17, top-1 prob at split points 0.17–0.60; tf 97.63% (noise floor 99.45%) | **Shipped** | [02](reports/02-int8-lm-head.md), [09](reports/09-precision-theory.md) |
| **Combo**: vLLM 0.30 + int4 draft head (C1), test card | P | +7.4% / +8.3% | +4.9% / +5.9% | +7.6% (long4k +8.7%) | +2.3% / +3.5% (decode +9.7%) | gate 2/17, split gap ≤0.25 on the baseline side, max 0.375 on the C1 side; probe (baseline : variant) 4 : 11 | **Shipped** | [04](reports/04-decode-six-items.md) |
| Combo, same-window production A/B | H8 | **+12.2% / +11.4%** | **+8.3% / +5.7%** | — | — | Soak: 319 requests, 0 errors, 0 Xid | **Shipped** (7 replicas) | [06](reports/06-canary-and-rollout.md) |
| **Gateway session affinity** (litellm session_affinity) | shuffle | — | — | — | Follow-up-turn TTFT mean −85% / −44%, median session time −63% / −40% (two rounds) | Does not change output | **Shipped** | [07](reports/07-session-affinity-routing.md) |

## Rejected (7 directions)

| Variant | Baseline | Single EN / ZH | c8 EN / ZH | long32k | Dev-agent c1 / c8 | Precision | Verdict | Report |
|---|---|---|---|---|---|---|---|---|
| 170hx-fullstack original setup (B) | A | +11.3% / −19.8% | −19.6% / −32.6% | −25.8% | +4.4% / −24.1% (decode −4.7%) | Theoretical quantization fidelity slightly higher (1–2 dB); uses a fine-tuned model, eval differences <1σ | Rejected (on this repo's workload and hardware) | [01](reports/01-three-way-comparison.md), [03](reports/03-dev-workload.md), [09](reports/09-precision-theory.md) |
| 170hx-fullstack inference stack + HyperQwen model (C′) | A | +18.0% / −12.2% | −19.3% / −28.9% | −23.0% | +10.0% / −15.2% (decode +6.8%) | Same weights as A, plus fp8 KV | Rejected (on this repo's workload and hardware) | [01](reports/01-three-way-comparison.md), [03](reports/03-dev-workload.md) |
| Truncated draft vocabulary 32k (V32) | P | −7.5% / −1.9% | −10.2% / −1.8% | −10.7% | −8.2% / −2.8% (decode −17.0%) | gate 2/17, splits ≤0.125 | Rejected | [04](reports/04-decode-six-items.md) |
| Truncated draft vocabulary 48k (V48) | P | −4.0% / +0.4% | −5.8% / +1.6% | −4.4% | −4.6% / +4.1% (decode −9.1%) | gate 1/17, splits ≤0.125 | Rejected | [04](reports/04-decode-six-items.md) |
| Truncated draft vocabulary 64k (V64) | P | −4.2% / +0.9% | −3.3% / +1.7% | −3.0% | −1.5% / +3.3% (decode −6.8%) | gate 1/17, splits ≤0.125 | Rejected | [04](reports/04-decode-six-items.md) |
| KV int8 (KI8) | P | −4.5% / −3.8% | −5.0% / +0.2% | −19.3% (TTFT 16→38 s) | −53.3% / −56.5% (decode −25.6%) | tf 99.46%, long4k 99.80% | Rejected | [08](reports/08-kv-cache-quantization.md) |
| KV fp8 (KF8) | P | −58.5% / −55.4% | −35.7% / −24.8% | −44.7% | −38.0% / −22.6% (decode −45.4%) | tf 99.20%, long4k 98.44% (below noise floor) | Rejected | [08](reports/08-kv-cache-quantization.md) |
| cpuset NUMA pinning (CS) | P | −0.6% / −0.9% | −0.5% / +0.6% | −0.1% | +0.1% / +2.9% | gate 17/17 | Rejected (within noise, no gain) | [04](reports/04-decode-six-items.md) |
| cpuset + nice −10 (CSN) | P | −0.7% / −0.6% | −0.9% / +0.9% | +0.1% | +0.0% / +5.4% | gate 17/17 | Rejected (within noise, no gain) | [04](reports/04-decode-six-items.md) |
| MTP + suffix, never (pipeline on, no override) | off | aggregate −5.4% / −3.7% | — | — | −3.9% / — (c4 −1.8%) | gate 17/17 | Diagnostic | [05](reports/05-mtp-suffix-hybrid.md) |
| MTP + suffix, τ=4 | off | aggregate +7.1% / +10.1% | +5.0% / +3.9% | — | **−11.7%** / −2.4% | gate 1/17, splits ≤0.125; tf 99.52% | Rejected | [05](reports/05-mtp-suffix-hybrid.md) |
| MTP + suffix, ema | off | aggregate +10.4% / 0.0% | +5.9% / +1.6% | — | **−11.9%** / −5.3% | gate 1/17, one at 0.25; tf 99.50% | Rejected | [05](reports/05-mtp-suffix-hybrid.md) |
| DFlash2 on vLLM sm80 | — | estimated +10~18% / −10~12% | estimated −15~−30% | — | — | Triggers Xid 31, not fixed upstream | Rejected (not run on card) | [10](reports/10-dflash2-and-hybrid-research.md) |

## Other sub-items tested

| Variant | Baseline | Single EN / ZH | c8 EN / ZH | long32k | Dev-agent c1 / c8 | Precision | Verdict | Report |
|---|---|---|---|---|---|---|---|---|
| Base model as shipped (D1: lm_head int8 + MTP int8) | A | −2.5% / −0.6% | −3.4% / −3.8% | −2.3% (long4k −8.7%) | — | gate 0/17, split points almost the same as D2 | Chose D2 (MTP stays int4) | [02](reports/02-int8-lm-head.md) |
| int4 draft head alone (D3) | P | +2.7% / +0.9% | +0.9% / +2.6% | +2.3% | +0.1% / +7.7% (decode +1.2%) | gate 2/17, splits ≤0.125; tf 99.48% | Merged into the combo (alone, most tiers within noise) | [04](reports/04-decode-six-items.md) |
| vLLM 0.29 alone (V029) | P | +1.4% / +1.6% | +2.0% / +2.1% | +1.4% | +0.7% / +4.1% (decode +5.5%) | gate 2/17, splits ≤0.125 | Small gain (mostly within noise) | [04](reports/04-decode-six-items.md) |
| vLLM 0.30 alone (V030) | P | +2.2% / +2.6% | +0.9% / +1.7% | +3.1% | −2.8% / +5.2% (decode −2.9%) | gate 0/17, splits ≤0.25 | Used as the combo's base | [04](reports/04-decode-six-items.md) |
| Baseline closing re-test (P3) | P | −0.5% / −0.9% | −0.1% / +1.8% | +0.5% | +0.1% / +4.2% | gate 17/17; tf 99.51% | Source of the noise band | [00](reports/00-methodology.md) |
| Full-vocabulary drafting vs upstream 40k draft vocabulary (2026-09-18) | 40k vocab | Chinese 68 → 127 tok/s | Chinese 210 → 419 | — | — | Does not change output | Adopted (already in the baseline) | [01](reports/01-three-way-comparison.md) §4 |
| `DRAFT_TOKENS` 6 vs 4 (2026-09-18) | k=4 | Flat | k=4 16–23% higher | — | — | Does not change output | Adopted k=4 | [01](reports/01-three-way-comparison.md) §4 |
| TP=2 across two cards (2026-09-18) | single card | 24–36 tok/s (6–9× slower) | 42 tok/s | — | — | — | Rejected | [01](reports/01-three-way-comparison.md) §4 |
| Power cap 250 W → 300 W | — | — | — | — | — | — | Not tested (the only hardware lever left) | [04](reports/04-decode-six-items.md) |

## Summary
- 3 shipped directions: int8 LM head, the combo (vLLM 0.30 + int4 draft head), gateway session affinity.
- 7 rejected directions: 170hx-fullstack (both forms), truncated draft vocabulary, KV fp8, KV int8, cpuset / nice, MTP + suffix hybrid, DFlash2 on vLLM sm80.
- The bottleneck across all tests: under full load the card sits at the 250 W power cap more than 95% of the time ([00](reports/00-methodology.md) §6).
