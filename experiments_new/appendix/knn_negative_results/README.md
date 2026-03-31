# KNN Imputation Negative Results

## Summary

KNN imputation exhibits **negative improvement** when combined with MIM, 
validating the hypothesis that MIM is most effective with simpler imputation methods.

## Key Findings

| Metric | Value |
|--------|-------|
| Negative Improvement Rate | 85.0% (17/20 MR settings) |
| Worst Case | -57.7% at MR=65% |
| Negative Cases Start | MR=15% |
| High MR (>60%) Negative Rate | 100% |

## Explanation

KNN produces high-quality imputations (Baseline MAE ≈ 0.016 at low MR), 
leaving minimal room for MIM's indicator mechanism to add value. 
The additional parameters from missing indicators introduce noise 
rather than useful information when the imputation is already accurate.

## Decision

KNN results are **excluded from main paper** and archived here as supplementary material.
Main paper focuses on Zero/Mean/Iterative imputation where MIM provides consistent positive improvements.
