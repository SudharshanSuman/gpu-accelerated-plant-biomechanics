# Benchmark — GPU-accelerated plant biomechanics

- solver: mode=`chain`, substeps=32, iterations=4
- 10 timed frames per point, 3 warm-up frames
- particles per plant: 25
- CUDA available during this run: True

| plants | particles | constraints | numpy (ms) | warp-cpu (ms) | warp-cuda (ms) | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 46 | 247.68 ± 0.44 | 1.74 ± 0.01 | 4.95 ± 0.02 | 0.4x |
| 16 | 400 | 736 | 269.31 ± 1.35 | 4.03 ± 0.01 | 5.26 ± 0.34 | 0.8x |
| 64 | 1600 | 2944 | 326.93 ± 1.33 | 11.22 ± 0.04 | 10.40 ± 0.65 | 1.1x |
| 256 | 6400 | 11776 | 576.13 ± 1.09 | 40.01 ± 0.08 | 27.73 ± 0.86 | 1.4x |
| 1024 | 25600 | 47104 | 1466.73 ± 1.40 | 156.29 ± 0.21 | 28.15 ± 1.07 | 5.6x |
| 4096 | 102400 | 188416 | — | 624.75 ± 0.84 | 30.54 ± 0.85 | 20.5x |
| 16384 | 409600 | 753664 | — | 2516.75 ± 2.92 | 116.04 ± 0.07 | 21.7x |