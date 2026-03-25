# Sweep & Inspection Summary — 2026-03-21

## The Key Discovery

**v1 ran on EPA only.** When we build EPA-specific P/Q distributions from the llama-v4 corpus, we exactly reproduce the v1 top words (branch, directive, clinton, phone, ruling, accessible, billing, protection, inquiries, telephone — ranks 1-18). The pooled cross-agency distribution buries these genuine style markers under domain jargon from CMS, NOAA, FAA etc.

## Results Matrix

| Experiment | Pre α (rule) | Post α (rule) | Ratio | Pre α (prop_rule) | Post α (prop_rule) | Ratio | Verdict |
|---|---|---|---|---|---|---|---|
| Unmatched, default, agency+quarter | 0.15% | 0.19% | 1.25x | 0.42% | 0.85% | 2.0x | Baseline |
| Cleaned corpus | 0.15% | 0.19% | 1.24x | 0.42% | 0.84% | 2.0x | No improvement |
| High thresholds (h=20,a=10) | 0.15% | 0.17% | 1.13x | 0.48% | 0.94% | 1.96x | Worse |
| Word length >= 4 | 0.11% | 0.12% | 1.10x | 0.17% | 0.25% | 1.48x | Kills signal |
| Word len4 + min250w | ~0% | ~0% | ~1x | ~0% | ~0% | ~1x | Kills everything |
| Filter short docs (250w) | 0.55% | 0.58% | 1.05x | 1.24% | 1.69% | 1.36x | Worse |
| Stratify by quarter only | ~0.1% | ~0.15% | ~1.5x | ~0.2% | ~0.4% | ~2.0x | Best so far |
| Per-agency pipeline | 0.54% | 0.77% | 1.42x | 0.97% | 1.61% | 1.65x | Noisy (small agencies) |
| Aggressive h=100,a=50 | 0.20% | 0.18% | 0.9x | 0.66% | 1.15% | 1.74x | Flat for rules |
| max_vocab=1000 | 0.49% | 0.55% | 1.1x | 0.84% | 1.22% | 1.45x | Modest |
| max_vocab=500 | 0.80% | 1.12% | 1.4x | 1.16% | 1.55% | 1.34x | Noisy |
| Frac-based 0.5% | 1.91% | 2.91% | 1.5x | 2.85% | 2.80% | 0.98x | Bad |
| LOR >= 0.3 | ~baseline | ~baseline | ~1.7x | - | - | - | No change |
| LOR >= 1.0 | 0.8% | - | - | - | - | - | Inflates noise |
| LOR >= 1.5 | 3.0% | - | - | - | - | - | Catastrophic |

## Key Findings from Inspections

### 1. Per-agency is the answer (04_rewrite_quality, EPA test)
- EPA-only distribution produces v1-quality results with genuine AI style markers
- Pooled distribution mixes domain jargon across 44 agencies, drowning out style signal
- Per-agency pipeline works for large agencies (EPA, FAA, CMS, SEC) but fails for small ones (noisy Q)
- **Hierarchical model is the right approach**: shrink small agencies toward pool, let large agencies use their own data

### 2. Sentence length asymmetry (09_sentence_level_scores)
- Human regulatory sentences average 12 words; AI sentences average 23 words
- This 2x length difference systematically inflates AI scores (more words = more negative LOR accumulated)
- PR Newswire doesn't have this problem (22 vs 23 words)
- AUC for regulatory text: 0.877 vs PR Newswire: 0.961

### 3. Prompt leakage exists but isn't the main issue (05_log_odds_analysis)
- 8.2% of docs contain "rewritten", but these are a tiny fraction of total vocabulary
- Cleaning barely changes aggregate statistics (mean LOR shifts by 0.007)
- The real problem is the 1.7-2.6x wider LOR distribution and systematic negative skew

### 4. Vocab filtering doesn't help (01, 02, 07, 08)
- Every filtering approach (word length, doc length, LOR threshold, count threshold, max vocab) kills signal proportionally to noise
- The "stabilizing" neutral-LOR words are actually important for the mixture model
- Aggressive filtering creates worse spurious alpha, not better

### 5. Coarser stratification helps (10_stratification)
- Quarter-only stratification reduces noise dramatically
- Weighted pre-ChatGPT alpha drops to ~0.1-0.2%
- The 2x post-ChatGPT ratio for proposed_rule is the most robust signal

### 6. False positives are agency-driven (06_false_positive_analysis)
- FEMA, FAA, NHTSA have highest spurious alpha
- Spearman r=0.568 between theoretical P(false positive) and actual pre-ChatGPT alpha
- Confirms agency-specific distributions are needed

## Recommended Next Steps

1. **Run hierarchical infer with optimal kappa on the cleaned corpus, stratified by quarter**
   - This combines the two biggest wins: agency-aware P + coarser stratification
2. **Run per-agency only for large agencies** (>500 AI docs: EPA, FAA, NOAA, FCC, FDA, IRS, AMS, FWS, CMS)
   - These have enough Q data for clean per-agency distributions
3. **Address sentence length asymmetry**
   - Consider normalizing log-probabilities by sentence length
   - Or subsample AI sentences to match human sentence length distribution
4. **Run the full pipeline on proposed_rule** (shows strongest signal across all experiments)
