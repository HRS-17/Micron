| Workload | Config | Hit rate | Avg latency (cyc) | Total cycles | Speedup | DRAM reads | PF accuracy | Late | Useless |
|---|---|---|---|---|---|---|---|---|---|
| LLM decode | No prefetch | 30.0% | 25.11 | 401,725 | 1.00x | 8,248 | - | 0 | 0 |
| LLM decode | Next-line | 86.9% | 6.33 | 180,522 | 2.23x | 14,431 | 52.1% | 7 | 6167 |
| LLM decode | Stride | 86.7% | 6.38 | 181,095 | 2.22x | 14,416 | 52.1% | 8 | 6152 |
| LLM decode | Stride + anti-pollution | 86.4% | 6.48 | 182,308 | 2.20x | 14,431 | 51.8% | 0 | 6179 |
| LLM decode | TAPS | 99.7% | 2.10 | 130,752 | 3.07x | 8,335 | 99.0% | 1 | 80 |
| LLM decode | TAPS + anti-pollution | 99.3% | 2.23 | 132,280 | 3.04x | 8,389 | 98.3% | 0 | 137 |
| CNN 3x3 conv | No prefetch | 51.0% | 18.17 | 264,357 | 1.00x | 4,768 | - | 0 | 0 |
| CNN 3x3 conv | Next-line | 97.3% | 2.90 | 115,725 | 2.28x | 5,824 | 81.0% | 0 | 1040 |
| CNN 3x3 conv | Stride | 97.2% | 2.92 | 115,923 | 2.28x | 5,824 | 81.0% | 0 | 1040 |
| CNN 3x3 conv | Stride + anti-pollution | 97.2% | 2.92 | 115,923 | 2.28x | 5,824 | 81.0% | 0 | 1048 |
| CNN 3x3 conv | TAPS | 99.1% | 2.30 | 109,917 | 2.41x | 5,056 | 95.5% | 0 | 208 |
| CNN 3x3 conv | TAPS + anti-pollution | 97.4% | 2.86 | 115,362 | 2.29x | 5,236 | 91.9% | 0 | 396 |
| Recommender | No prefetch | 4.7% | 33.44 | 154,841 | 1.00x | 3,476 | - | 0 | 0 |
| Recommender | Next-line | 73.6% | 10.72 | 71,945 | 2.15x | 7,301 | 41.0% | 0 | 3735 |
| Recommender | Stride | 48.2% | 19.11 | 102,536 | 1.51x | 3,500 | 98.5% | 0 | 20 |
| Recommender | Stride + anti-pollution | 47.5% | 19.31 | 103,295 | 1.50x | 3,523 | 97.1% | 0 | 43 |
| Recommender | TAPS | 73.8% | 10.66 | 71,714 | 2.16x | 3,496 | 99.2% | 0 | 16 |
| Recommender | TAPS + anti-pollution | 73.1% | 10.87 | 72,473 | 2.14x | 3,519 | 98.3% | 0 | 39 |
