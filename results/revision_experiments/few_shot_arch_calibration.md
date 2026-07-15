# Few-Shot Architecture Calibration

## Motivation
LOAO (0-shot) error is ~45% overall, reaching 96% for the `equiv` family.
When a new architecture family is encountered, profiling k members significantly reduces error.

## Summary Table

| Family | N Members | 0-shot MAPE | 1-shot MAPE | 2-shot MAPE | Improvement |
| --- | --- | --- | --- | --- | --- |
| CNN | 2 | 27.4% | 15.1% | — | 12.3pp |
| TCN | 1 | 14.0% | — | — | — |
| TCN-NAS | 1 | 53.3% | — | — | — |
| equiv | 1 | 47.4% | — | — | — |
| mobile | 4 | 19.2% | 16.5% | 13.5% | 2.7pp |
