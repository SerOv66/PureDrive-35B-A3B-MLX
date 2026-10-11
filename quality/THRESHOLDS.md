# THRESHOLDS — quality-gate thresholds

Read by `quality/compare.py` (the fenced JSON block below). In the original project this file was frozen together with the
quality suite; changing a threshold after the freeze required re-measuring every configuration plus auditor approval.
(This is an English translation; the JSON block is unchanged.)

Usage: `python quality/compare.py ref.json cand.json --profile {lossless|numeric}`

## Thresholds (machine-readable)

```json
{
  "version": 1,
  "profiles": {
    "lossless": {
      "identity":  {"min_identical": 6},
      "coding":    {"max_drop": 0, "exact": true},
      "reasoning": {"max_drop": 0, "exact": true},
      "tools":     {"max_drop": 0, "exact": true},
      "needle":    {"max_lost": 0}
    },
    "numeric": {
      "identity":  {"informational": true},
      "coding":    {"max_drop": 2},
      "reasoning": {"max_drop": 2},
      "tools":     {"max_drop": 1},
      "needle":    {"max_lost": 0}
    }
  }
}
```

Meaning: `max_drop` = number of items (problems / questions / cases) the candidate may lose relative to the reference
(coding = sum of HumanEval+ and MBPP+ problems passed with base+plus tests); `exact` = the count must equal the reference;
`min_identical` = token-identical identity prompts (out of 6); `max_lost` = needle cells found by the reference and lost by
the candidate. A part missing from either file is `SKIP` (does not fail): check the table. Reference and candidate must use
the same `--suite` (otherwise `compare.py` exits with code 2).

## Profile `lossless` — speculative decoding, MTP, kernel/layout/scheduling changes that must be exact

- **identity = 6/6 IDENTICAL.** At temperature 0 the output must be token-identical to the baseline without the
  optimization. A single divergence is a bug (precedent: EXP-005, multi-row verify with a different reduction order diverged
  at the 4th-5th token). Comparison is on the generated text (256 tokens, 6 prompts): identical text <=> identical tokens
  (detokenization is deterministic).
- **coding / reasoning / tools = score equal to the reference** (`exact`). If the output is token-identical the scores are too;
  a score difference with identity 6/6 indicates server non-determinism or a harness bug to investigate, not to absorb.
- **needle = no cell lost.** Needle is the test that first catches length-dependent errors (KV, split-K attention, RoPE).

## Profile `numeric` — numerically different but legitimate kernels/runtimes

Applicable to: runtime comparisons, kernels with a different reduction order, layout conversions, full-precision KV with a
different attention implementation. NOT applicable to weight re-quantization (forbidden) or quantized KV (needs its own
KL thresholds).

- **coding >= ref - 2 problems, reasoning (GSM8K) >= ref - 2 questions, tools >= ref - 1 case.** With greedy decoding a kernel
  that sums in a different order flips a few near-tie tokens; the trajectory changes but the capability does not. On the quick
  subset (57 coding problems, 30 GSM8K, 14 tool cases) -2/-1 items corresponds to about 3.5% / 6.7% / 7%: tighter than the
  expected noise of a lucky flip, wide enough to avoid false FAILs from a single diverging trajectory.
- **needle = every cell found by the reference must be found.** No tolerance: a lost needle at 64K/128K is the typical symptom
  of a long-context regression.
- **identity = informational only**: different numerics are not token-identical by construction; the number of identical
  prompts and the first-divergence position are recorded but do not decide.

## Calibration notes

- The `max_drop` thresholds apply to `quick` and `full`; on `full` (541 coding problems) -2 is very tight: if two runs of the
  SAME runtime with different seeds already differ by more, the noise is larger than expected and the threshold must be
  justified and re-measured (a documented decision, not a silent change).
- The reference must be measured with the same suite and the same harness (`harness.sha256` in the JSON) as the candidate.

## Not covered here — perplexity / KL divergence

KL divergence of the custom attention kernel against the stock attention was measured separately in-process
(`quality/kl_decode.py`, results in `quality/results/kl_*.json`), not by this suite. The KV cache stays bf16, so no KV-quantization
thresholds are defined.
