### 3.3 Sampling Strategy Comparison

**Table 1a. Sampler comparison (30 steps)**

| Model/Sampler      | Schedule              |   Latency (s/image) |   FID ↓ |   CLIP Score ↑ |
|:-------------------|:----------------------|--------------------:|--------:|---------------:|
| SD v1.5 / DDIM     | Native DDIM / leading |               5.434 | 151.328 |         31.14  |
| SD v1.5 / Euler    | Non-Karras / linspace |               5.424 | 152.114 |         31.22  |
| SD v1.5 / DPM++ 2M | Karras                |               5.435 | 150.11  |         31.331 |

**Table 1b. Solver/schedule ablation (30 steps)**

| Model/Sampler      | Schedule              |   Latency (s/image) |   FID ↓ |   CLIP Score ↑ |
|:-------------------|:----------------------|--------------------:|--------:|---------------:|
| SD v1.5 / Euler    | Non-Karras / linspace |               5.424 | 152.114 |         31.22  |
| SD v1.5 / Euler    | Karras                |               5.425 | 151.837 |         31.257 |
| SD v1.5 / DPM++ 2M | Non-Karras / linspace |               5.348 | 150.44  |         31.178 |
| SD v1.5 / DPM++ 2M | Karras                |               5.435 | 150.11  |         31.331 |
