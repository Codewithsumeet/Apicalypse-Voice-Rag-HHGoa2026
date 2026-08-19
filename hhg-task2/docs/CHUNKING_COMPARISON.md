# Chunking Strategy Comparison

**Test Queries:** 20


## Results

| Strategy | Vectors | Avg Latency | P50 Latency | P100 Latency | Avg Top Score | Min Top Score |
|---|---|---|---|---|---|---|
| fixed | 11,260 | 50.6ms | 49.7ms | 68.8ms | 0.6124 | 0.3808 |
| semantic | 12,150 | 56.6ms | 55.8ms | 70.2ms | 0.6351 | 0.4503 |
| metadata_aware | 11,217 | 52.9ms | 51.9ms | 70.5ms | 0.6103 | 0.3808 |

## Recommendation

**Winner:** `semantic` — highest average relevance score (0.6351)
