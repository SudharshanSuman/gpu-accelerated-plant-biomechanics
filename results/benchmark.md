# Benchmark — GPU-accelerated plant biomechanics

- solver: mode=`chain`, substeps=32, iterations=4
- 6 timed frames per point, 3 warm-up frames
- particles per plant: 25
- CUDA available during this run: False

| plants | particles | constraints | numpy (ms) | warp-cpu (ms) | warp-cuda (ms) | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 46 | 282.14 ± 5.72 | 1.89 ± 0.11 | — | — |
| 16 | 400 | 736 | 309.56 ± 7.52 | 4.48 ± 0.21 | — | — |
| 64 | 1600 | 2944 | 371.20 ± 9.22 | 13.19 ± 0.24 | — | — |

> The CUDA column is empty because no NVIDIA driver was reachable when
> this table was generated (`nvidia-smi` could not talk to the driver).
> The plot therefore shows the CPU curve only. Re-run this script on the
> RTX 4070 to fill in the GPU column; nothing else needs to change.