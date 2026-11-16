English translation of c3_gate_v1.md (translated 2026-11-16). The Chinese original is authoritative. 中文原文：c3_gate_v1.md

This document was frozen before results were seen; the translation was made afterwards and does not change the frozen content.

# C3 gate rule v1 (c3-gate-v1; frozen 2026-11-05, written before the GPU job was submitted and not changed after the results were seen)

## Data and conditions
- Mixture samples: `configs/mix_v1_devel100_v2.json`, 100 groups × 2 swapped samples, audio in `data/mix_v1/devel_seed2026/g*/mic0_16k.wav` (job 22013065).
- Input of C3: the mic 0 single-channel mixture signal (16 kHz); the text is the frozen direction instruction (`docs/mix_spec_v1.md`, θ1 rounded to whole degrees), followed by the frozen P2 prompt.
- Target of each swapped sample: for s1 it is A (A at θ1), for s2 it is B (B at θ1); the interferer is the other person of the same group.
- Scoring: slu-score-v2 (`docs/scoring_rules.md`) and binding-score-v1 (`docs/binding_score_v1.md`), both unchanged.
  - D_T = target − interferer, D_I = interferer − target, computed on the multisets of entity values after normalize_v2.
  - Only the parsed entities are considered, matched as complete values. On parse failure, entities are taken as the empty set.

## Definitions
- Each sample is assigned by binding-score-v1 to one of four classes: target only, interferer only, both, neither.
- **Single-talker verdict**: samples classed as "target only" or "interferer only".
- **Target share** = number of target-only samples / number of samples with a single-talker verdict.
- **95% confidence interval**: bootstrap **with the pair (group) as the unit**. Draw 100 groups with replacement from the 100 groups, with the two swapped samples of each group entering together; repeat 10,000 times, `numpy.random.default_rng(2026)`; recompute the target share on each resample (skip a resample in which there are 0 single-talker verdicts, and record how many times this happens); take the 2.5% and 97.5% quantiles as the interval. The pair rather than the speaker is chosen as the unit because the same speaker appears in several groups, and the two swapped samples of each group use the same pair of recordings and are not independent of each other.

## Gate rules (judged in order)
1. Fewer than 50 samples with a single-talker verdict: **stop and report**; too few samples to judge.
2. Lower bound of the confidence interval > 0.60: **stop**; leakage suspected.
3. Upper bound of the confidence interval < 0.40: **stop**; the model may be biased towards the interferer, or the data may have a problem.
4. Otherwise: **pass**; may proceed to front-end inference.

The thresholds are empirical and are not changed after the results are seen.

## Also reported (not part of the gate)
- The four class proportions; target-any and all target values hit (per sample).
- **Pair-flip accuracy** = number of groups in which both swapped samples are classed "target only" / 100. That is, after the azimuths are swapped, the model's answer also switches to the other person.
- **Whether the answer changes between the swaps**:
  - (a) compared by "whom the answer points to": each sample's output is classed as A only, B only, both, neither; if the two swapped samples are classed differently, it counts as "changed".
  - (b) whether the multisets of entity values parsed from the two swapped samples (after normalize_v2) are exactly identical.
- Number of parse failures, number of refusals (`scripts/qwen/postproc_c1_ext.py: REFUSAL`), number of out-of-inventory labels.
- Stratified exploration (reference only, not part of the gate):
  - angle: four strata, [0,15), [15,30), [30,60), [60,180];
  - T60: [0.2,0.3), [0.3,0.4), [0.4,0.5];
  - active-frame overlap ratio: [0,0.5), [0.5,0.75), [0.75,0.9), [0.9,1];
  - SIR of the full image: < −0.5, [−0.5,0.5], > 0.5 dB.
  - Overlap ratio and SIR are per-sample recorded values from the manifest of the generation job.
- C1 (same job, clean recordings, P2 prompt):
  - for the 124 recordings involved, compute whether the target values are hit against each opponent in each group;
  - compare with the P2 results of job 22011390 on the 50 overlapping recordings: whether the raw outputs are identical and whether the parsed entities are identical.

## Not done
The separation front end is not run; the prompt, scoring and gate rules are not changed according to results.
