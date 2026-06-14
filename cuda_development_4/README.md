# Reduce fp64 pressure, and Particle reordering

## Pull Request 16

https://github.com/stasostyk/DSMC/pull/16

Currently, we use doubles (64 bit precision) for everything. We do not need it for the particle's position and velocity, 32 bit precision is perfectly fine for the state there. We just need double for the intermediate calculations. Therefore, here we change only particle state (pos and vel) to floats and some of the boundary calculations (since position is in floats now), and everything else remains in double precision. 

Although a naive optimization, on a problem size of 20x20x20, we go from 14843.1379 ms to 12010.0141 ms, so about a **23.5% speedup.** 

More than just being less arithmetic, many of our kernels (especially NTC collisions) struggled with register spilling and of fp64 pressure from so many heavy computations (this was revealed by NSight Compute which recommended reducing fp64 operations). 


## Pull Request 20

https://github.com/stasostyk/DSMC/pull/20

Memory coalescing was a big problem in the DSMC run. Warps tend to execute within particles on a cell, but as a grid-free solver, DSMC particles travel across cells, so NSight compute showed kernels having around 60% memory accesses be uncoalesced. 

Using the CUDA CUB parallel primatives library for an efficient implementation of prefix sum (https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html), offsets are computed for each cell count. Based off of that, an efficient one-thread-per-particle reordering is done (after the indexing stage). 

I experimented with how often to call this reordering. With many tests from every 500 steps to every step, I found that exactly every 20 steps is optimal to call the reordering. This is because particles slowly drift due to collisions, but not fast enough to necessitate reordering every iteration. The theoretical max is fascinating, because it is the only one to DRASTICALLY reduce the collision scheme; it seems that NTC gets faster only with perfectly ordered particles, anything a bit off and boom it doesnt gain any speedup. This could be looked into to leverage the particle order more in the collision scheme. 

| Row | Baseline time (ms) | Reorder every 20 iters time (ms) | Speedup vs baseline | Reorder every iter time (ms) | Speedup vs baseline |
|---|---:|---:|---:|---:|---:|
| Entire program | 33355.3465 | 31954.7338 | **4.20%** | 39127.0298 | -17.30% |
| no_time_counter_scheme_kernel | 13367.3636 | 13028.0362 | 2.54% | 10401.1746 | **22.19%** |
| move_particles_kernel | 5436.5715 | 5434.3626 | 0.04% | 5390.4085 | 0.85% |
| bin_particles_kernel | 4820.8184 | 2178.6170 | 54.81% | 1580.4815 | 67.22% |
| generate_particles_in_rect_kernel | 3011.4669 | 3011.1036 | 0.01% | 3010.0364 | 0.05% |
| filter_particles_out_of_bounds | 2119.0388 | 2158.6138 | -1.87% | 2184.9309 | -3.11% |
| accumulate_sampling_kernel | 1863.8886 | 1815.2524 | 2.61% | 1684.7101 | 9.61% |
| hss_scheme_kernel | 525.7128 | 455.7363 | 13.31% | 403.8196 | 23.19% |
| reorder_particles_by_cell_kernel | — | 1657.5582 | N/A | 12178.4093 | N/A |
| cub::DeviceScanKernel | — | 0.4610 | N/A | 11.7402 | N/A |
| cub::DeviceScanInitKernel | — | 0.2038 | N/A | 5.0742 | N/A |

Other minor improvements _(made sense although achieved 0% speedup xd)_:
- No need to check for particle out of bounds in binning step since we already filter them out in the filter out of bounds step just before binning
- Replaced the reset to zero kernel with a cudaMemset

