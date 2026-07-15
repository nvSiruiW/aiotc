# Deployment Selector Oracle Evaluation

## Protocol
- **Selector**: uses cost model (N_exec + B_eff linear regression) to predict latency.
- **Oracle**: exhaustive search over real measurements; picks feasible config with best ATE.
- **Regret**: ATE/latency/energy difference between selector and oracle.
- **Feasible**: selector's chosen config satisfies deadline with real (measured) latency.

## Aggregate Summary

### All scenarios with oracle
- Total scenarios (oracle exists): 44
- Selector produced a prediction: 44
- Feasible configurations: 44/44 (feasible rate = 100.0%)
- Top-1 match (selector = oracle): 44/44 (rate = 100.0%)
- Median ATE regret: 0.0000 m
- Mean ATE regret: 0.0000 m
- Worst-case ATE regret: 0.0000 m
- Median latency regret: 0.000 ms

### Per-device summary

| Device | Scenarios | Feasible Rate | Top-1 Rate | Median ATE Regret | Max ATE Regret |
| --- | --- | --- | --- | --- | --- |
| blackwell | 30 | 100.0% | 100.0% | 0.0000 m | 0.0000 m |
| agx_orin | 7 | 100.0% | 100.0% | 0.0000 m | 0.0000 m |
| orin_nano | 7 | 100.0% | 100.0% | 0.0000 m | 0.0000 m |

## Per-Scenario Detail (first 30 rows shown)

| Device | Deadline | ATE≤ | Energy Budget | Predicted | Oracle | Pred Lat | True Lat | Oracle Lat | ATE Regret | Feasible | Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blackwell | 1.0 ms | 0.9 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 1.0 ms | 1.0 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 1.0 ms | 1.1 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 1.0 ms | 1.2 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 2.0 ms | 0.9 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 2.0 ms | 1.0 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 2.0 ms | 1.1 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 2.0 ms | 1.2 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 0.9 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 5.0 ms | 1.0 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.1 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.2 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 0.9 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 10.0 ms | 1.0 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.1 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.2 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 20.0 ms | 0.9 m | none | None/None | None/None | None | None | None | None | ✗ | ✗ |
| blackwell | 20.0 ms | 1.0 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 20.0 ms | 1.1 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 20.0 ms | 1.2 m | none | ronin_resnet18/fp16 | ronin_resnet18/fp16 | 1.368 | 1.502 | 1.502 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.0 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.1 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.0 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.1 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 20.0 ms | 1.0 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 20.0 ms | 1.1 m | tight | imunet/fp32 | imunet/fp32 | 1.677 | 1.804 | 1.804 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.0 m | medium | mnasnet/fp32 | mnasnet/fp32 | 1.885 | 2.314 | 2.314 | 0.0 | ✓ | ✓ |
| blackwell | 5.0 ms | 1.1 m | medium | mnasnet/fp32 | mnasnet/fp32 | 1.885 | 2.314 | 2.314 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.0 m | medium | mnasnet/fp32 | mnasnet/fp32 | 1.885 | 2.314 | 2.314 | 0.0 | ✓ | ✓ |
| blackwell | 10.0 ms | 1.1 m | medium | mnasnet/fp32 | mnasnet/fp32 | 1.885 | 2.314 | 2.314 | 0.0 | ✓ | ✓ |

## Paper Summary Sentence

> Across 44 deployment scenarios (3 devices × 5 deadlines × 4 accuracy constraints + energy variants), the selector produced feasible configurations in 44/44 cases (100.0%) and achieved median ATE regret of 0.0000 m compared with exhaustive oracle search.
