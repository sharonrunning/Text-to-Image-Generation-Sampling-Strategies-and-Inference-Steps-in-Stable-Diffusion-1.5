### 3.4 Hyperparameter Analysis

**Table 2. Step-count ablation (DPM++ 2M Karras)**

|   Steps |   Latency (s/image) |   Same-session latency (s/image) |   FID ↓ |   CLIP Score ↑ |
|--------:|--------------------:|---------------------------------:|--------:|---------------:|
|      10 |               1.728 |                            1.658 | 151.08  |         31.029 |
|      20 |               3.28  |                            3.078 | 151.437 |         31.31  |
|      30 |               5.435 |                            4.505 | 150.11  |         31.331 |
|      40 |               6.345 |                            5.966 | 151.592 |         31.256 |
|      50 |               7.878 |                            7.552 | 150.216 |         31.265 |

Latency (s/image) is the mean of the 200 logged generation times. The 30-step batch was logged in Member A's T4 session and the 10/20/40/50-step batches in Member B's, so comparisons against the 30-step row mix session and step-count effects. Same-session latency re-times all five step counts in one T4 session (first 20 prompts, step counts interleaved, images discarded); use it for speedups across step counts.
