# Hybrid AoS/SoA memory layout, Work Queues for faster collisions, Adding MPI support for multiple GPUs

## Pull Request 23

https://github.com/stasostyk/DSMC/pull/23

As a sort of hybrid mix between SoA and AoS, this optimizes coalesced memory access by leveraging memory banking. CUDA banks are typically four bytes so we still dont fit in a single bank, but the accesses are far more coalesced. I initially thought that it would retrieve nicely in chunks of 16 bytes, so I added padding. However turns out that this just inflates memory IO time, because after removing the padding (so we have 12 byte values packed one after another), it got even faster! Below is the summarizing the results:

Overall 16.29% improvement!

Speedup formula: (baseline time / optimized time - 1) × 100. Positive means faster than baseline.

| Metric | Baseline | Pack with padding | Pack without padding |
|---|---:|---:|---:|
| Initialization elapsed time | 666.6370 ms | 672.6870 ms (-0.90%) | 685.6178 ms (-2.77%) |
| Simulation loop elapsed time | 26711.7594 ms | 24574.0277 ms (+8.70%) | 22714.2218 ms (+17.60%) |
| **All program elapsed time** | 28408.7264 ms | 26312.3658 ms (+7.97%) | 24429.8763 ms **(+16.29%)** |
| CUDA API: cudaMemcpy total time | 26892.9608 ms | 24808.7988 ms (+8.40%) | 22900.8423 ms (+17.43%) |
| Kernel: no_time_counter_scheme total time | 7999.3111 ms | 3826.5423 ms (+109.05%) | 3791.9530 ms (+110.95%) |
| Kernel: move_particles total time | 5502.4967 ms | 6012.7937 ms (-8.49%) | 5658.5132 ms (-2.76%) |
| Kernel: counting_sort_scatter total time | 4297.5714 ms | 4298.4206 ms (-0.02%) | 3854.2668 ms (+11.50%) |
| Kernel: scatter_and_key total time | 2674.6308 ms | 3847.1743 ms (-30.48%) | 3211.3467 ms (-16.71%) |
| Kernel: accumulate_sampling total time | 1253.8859 ms | 741.5385 ms (+69.09%) | 806.2923 ms (+55.51%) |

NOTE: After integrating MPI, the speedup is closer to overall 7.8% because of the MPI degredation to performance even on single GPU


## Pull Request 24

https://github.com/stasostyk/DSMC/pull/24

### Optimizations
* Collision kernel changed: instead of each thread takes one cell, now a lot of threads are spawned, and they keep taking cells to process from the queue.  In the queue the cells are sorted by their cell counts in decreasing order.
* In Move particles kernel, before checking for ray-sphere intersection, firstly we check if the distance is large enough

### Comparison

#### Setup
```
#define MAX_PARTICLES 30000000
#define MAX_PARTICLES_PER_CELL 750
#define NX 50
#define NY 50
#define NZ 50

config->nSteps = 2000;
config->particlesPerCellTarget = 100;

./DSMC BALL
```

#### Improvement
|   |   BEFORE | AFTER   |   improvement |
|---|---|---|---|
| Simulation loop  |  78677.7596  ms   | 50394.6421 ms  | ~**56.1% speedup** |
| Kernel for No Time Collision Scheme (total time) |  22,310,262,796 ns   | (work queue implementation) 8,357,774,082  ns  | ~**2.7x speedup** |
|  `move_particles_kernel` (total time) |  18,560,268,551 ns   |  4,016,009,796  | ~**4.6x speedup** |


#### Stats before (main branch)
```
/content/DSMC/cuda/build
Running case: BALL
INITIALIZATION. Elapsed time: 1347.9644 ms
SIMULATION LOOP. Elapsed time: 78677.7596 ms
step=2000
  NP=13231887
  mean_u=(9.764032e+02, -4.125858e+00, 1.004245e-01)
  T=3.382271e+02
  totalCollisions = 1284062517
ALL PROGRAM FINISHED. Elapsed time: 81198.4435 ms
Generating '/tmp/nsys-report-eb22.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/DSMC/cuda/build/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ----------------------
     99.7   81,114,703,802        818  99,162,229.6  100,141,581.0     1,678  345,838,600  13,922,076.3  poll                  
      0.2      180,873,293        652     277,413.0       15,733.0       318   18,358,105     910,886.2  ioctl                 
      0.0       38,453,537          2  19,226,768.5   19,226,768.5   826,465   37,627,072  26,021,958.8  pthread_rwlock_wrlock 
      0.0        1,803,680         29      62,195.9       11,265.0     8,770    1,207,129     221,232.0  mmap64                
      0.0          822,145          1     822,145.0      822,145.0   822,145      822,145           0.0  pthread_cond_wait     
      0.0          526,709         12      43,892.4       17,914.5     1,171      344,352      95,023.3  write                 
      0.0          512,757         10      51,275.7       53,479.0    28,718       72,304      15,162.1  sem_timedwait         
      0.0          387,116         47       8,236.5        7,628.0     2,062       26,531       4,264.9  open64                
      0.0          362,410         31      11,690.6        3,741.0     2,020      170,108      30,037.5  mmap                  
      0.0          333,734         38       8,782.5        3,095.0     1,125       92,969      16,768.8  fopen                 
      0.0          168,699         38       4,439.4        3,700.5     2,455       13,055       2,285.8  munmap                
      0.0          164,211      2,550          64.4           54.0        35        2,258          60.2  fputc                 
      0.0          117,179          2      58,589.5       58,589.5    52,363       64,816       8,805.6  pthread_create        
      0.0           68,845         31       2,220.8        1,169.0       832       15,107       2,915.1  fclose                
      0.0           49,892         15       3,326.1          145.0        45       36,082       9,231.6  fwrite                
      0.0           37,653         20       1,882.7           45.0        44       36,709       8,197.3  fgets                 
      0.0           33,056         62         533.2          554.5       166        1,220         204.7  fcntl                 
      0.0           30,274          1      30,274.0       30,274.0    30,274       30,274           0.0  connect               
      0.0           28,041          6       4,673.5        5,291.5     1,515        6,948       2,075.4  open                  
      0.0           23,431         15       1,562.1        1,434.0       972        3,396         641.5  read                  
      0.0           18,681          2       9,340.5        9,340.5     7,113       11,568       3,150.2  socket                
      0.0           17,621          3       5,873.7        7,120.0     2,820        7,681       2,659.4  pipe2                 
      0.0            7,134          2       3,567.0        3,567.0     1,807        5,327       2,489.0  pthread_cond_broadcast
      0.0            2,909          8         363.6          360.0       273          439          47.6  dup                   
      0.0            1,487          1       1,487.0        1,487.0     1,487        1,487           0.0  bind                  
      0.0            1,007          1       1,007.0        1,007.0     1,007        1,007           0.0  listen                
      0.0              545         10          54.5           37.5        35          187          47.1  fflush                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     99.4   79,379,910,944      6,009   13,210,169.9       24,812.0        7,763  790,383,884  21,326,966.4  cudaMemcpy            
      0.2      195,968,897     32,103        6,104.4        5,113.0        3,356      433,463       4,752.3  cudaLaunchKernel      
      0.2      195,452,424          1  195,452,424.0  195,452,424.0  195,452,424  195,452,424           0.0  cudaMemcpyToSymbol    
      0.1       40,698,001      2,003       20,318.5       17,358.0        6,261      869,428      20,689.4  cudaMemset            
      0.0       24,167,984         24    1,006,999.3    1,039,560.5        4,260    2,428,319     685,770.2  cudaFree              
      0.0        1,906,522         24       79,438.4       78,340.5        3,463      176,197      49,092.5  cudaMalloc            
      0.0            1,684          1        1,684.0        1,684.0        1,684        1,684           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------------------------------------------------------------------------------------
     28.1   22,310,262,796      2,000   11,155,131.4   11,232,310.0    8,767,143   11,431,739    353,824.1  no_time_counter_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)   
     23.4   18,560,268,551      2,000    9,280,134.3    9,294,501.0    8,707,623    9,412,439    104,779.6  move_particles_kernel(Particles, int, curandStateXORWOW *)                                          
     15.4   12,217,086,973      2,000    6,108,543.5    6,044,255.5    5,466,173  107,948,347  2,283,541.7  counting_sort_scatter_kernel(Particles, Particles, int *, int *, int)                               
     12.2    9,659,253,796      2,000    4,829,626.9    4,808,249.0    3,738,791    5,187,822    163,594.0  scatter_and_key_kernel(Particles, Particles, int *, int *, int *, int *, int)                       
     10.8    8,556,114,794     12,001      712,950.2      715,577.0      658,522    6,776,372     57,333.2  generate_particles_in_rect_kernel(Particles, int, int, float, float, float, float, float, float, in…
      3.5    2,765,319,421        100   27,653,194.2   27,628,963.0   27,013,331   28,448,809    266,041.3  accumulate_sampling_kernel(int *, int *, Cell *, Particles)                                         
      3.1    2,457,284,211      2,000    1,228,642.1    1,227,381.5    1,166,584    1,260,020     13,928.6  mark_valid_kernel(Particles, int *, int)                                                            
      1.7    1,385,059,739      4,000      346,264.9      333,470.0        6,175      699,643    339,160.9  void cub::CUB_200700_750_NS::DeviceScanKernel<cub::CUB_200700_750_NS::DeviceScanPolicy<int, cub::CU…
      1.0      780,744,728          1  780,744,728.0  780,744,728.0  780,744,728  780,744,728          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.7      576,886,900      2,000      288,443.5      286,894.0      163,871      356,637     25,001.2  hss_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)               
      0.0        7,574,007      4,000        1,893.5        1,888.0        1,600        2,240        131.5  void cub::CUB_200700_750_NS::DeviceScanInitKernel<cub::CUB_200700_750_NS::ScanTileState<int, (bool)…
      0.0        2,895,661          1    2,895,661.0    2,895,661.0    2,895,661    2,895,661          0.0  filter_particles_inside_ball(Particles, Particles, int, int *)                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  ----------  -----------  ------------------
     94.6      210,059,072  4,009  52,396.9   1,344.0     1,216  33,986,717  1,304,668.8  [CUDA memcpy DtoH]
      3.1        6,795,140  2,000   3,397.6   3,360.0     2,816       4,000        265.0  [CUDA memcpy DtoD]
      2.3        5,093,934  2,003   2,543.2   2,528.0       448      17,792        388.9  [CUDA memset]     
      0.0              704      1     704.0     704.0       704         704          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
  1,005.000  2,003     0.502     0.500     0.000     5.000        0.102  [CUDA memset]     
  1,000.000  2,000     0.500     0.500     0.500     0.500        0.000  [CUDA memcpy DtoD]
    322.581  4,009     0.080     0.000     0.000    52.928        2.048  [CUDA memcpy DtoH]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]

Generated:
    /content/DSMC/cuda/build/report1.nsys-rep
    /content/DSMC/cuda/build/report1.sqlite

```

#### Stats now (this branch)
```
/content/DSMC/cuda/build
Running case: BALL
INITIALIZATION. Elapsed time: 1278.9171 ms
SIMULATION LOOP. Elapsed time: 50394.6421 ms
step=2000
  NP=13228939
  mean_u=(9.763676e+02, -3.692221e+00, 1.029485e-01)
  T=3.385939e+02
  totalCollisions = 1283749846
ALL PROGRAM FINISHED. Elapsed time: 52846.3935 ms
Generating '/tmp/nsys-report-c949.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/DSMC/cuda/build/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ----------------------
     99.6   52,760,357,021        535  98,617,489.8  100,149,994.0   343,790  347,088,853  17,214,797.5  poll                  
      0.3      183,718,159        659     278,783.2       14,858.0       318   17,627,151     897,155.8  ioctl                 
      0.0       19,626,439          4   4,906,609.8      458,654.5    50,216   18,658,914   9,171,525.6  pthread_rwlock_wrlock 
      0.0        1,772,718         29      61,128.2       11,200.0     8,913    1,234,654     226,289.7  mmap64                
      0.0        1,744,569         10     174,456.9       57,098.0    27,448      597,423     215,026.1  sem_timedwait         
      0.0          512,822          1     512,822.0      512,822.0   512,822      512,822           0.0  pthread_cond_wait     
      0.0          413,958         47       8,807.6        7,822.0     1,920       21,744       4,308.1  open64                
      0.0          380,614         38      10,016.2        3,806.5     1,376       98,635      17,964.1  fopen                 
      0.0          233,955         15      15,597.0          791.0        42      200,769      51,487.8  fwrite                
      0.0          223,271         12      18,605.9       18,582.5     3,543       28,349       5,961.6  write                 
      0.0          221,868         31       7,157.0        3,466.0     2,024       62,162      11,204.6  mmap                  
      0.0          179,556          2      89,778.0       89,778.0    68,745      110,811      29,745.2  pthread_create        
      0.0          175,384      2,550          68.8           53.0        35        5,391         113.9  fputc                 
      0.0          174,565         38       4,593.8        4,064.0     2,331       21,362       3,093.7  munmap                
      0.0           79,037         31       2,549.6        1,594.0       798       16,221       3,277.1  fclose                
      0.0           38,053         20       1,902.7           45.0        44       37,132       8,292.1  fgets                 
      0.0           31,433         62         507.0          515.5       162        1,412         202.9  fcntl                 
      0.0           29,598          6       4,933.0        4,952.0     1,528        7,507       2,152.4  open                  
      0.0           29,515         15       1,967.7        1,464.0     1,244        5,859       1,248.1  read                  
      0.0           24,825          2      12,412.5       12,412.5     9,042       15,783       4,766.6  socket                
      0.0           16,103          1      16,103.0       16,103.0    16,103       16,103           0.0  connect               
      0.0           14,502          3       4,834.0        5,716.0     2,350        6,436       2,181.1  pipe2                 
      0.0            7,239          2       3,619.5        3,619.5     2,258        4,981       1,925.5  pthread_cond_broadcast
      0.0            2,800          8         350.0          365.0       251          432          62.7  dup                   
      0.0            1,922          1       1,922.0        1,922.0     1,922        1,922           0.0  bind                  
      0.0            1,622          1       1,622.0        1,622.0     1,622        1,622           0.0  listen                
      0.0              687         10          68.7           36.0        35          362         103.1  fflush                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     80.1   41,127,830,479      8,009    5,135,201.7       24,593.0        7,099  743,055,125  12,409,858.7  cudaMemcpy            
     18.3    9,419,314,849      4,000    2,354,828.7    1,878,679.0        1,444    7,058,033   2,354,588.1  cudaStreamSynchronize 
      0.6      304,651,855     46,103        6,608.1        5,631.0        3,329    2,058,109      13,397.3  cudaLaunchKernel      
      0.3      174,718,992          1  174,718,992.0  174,718,992.0  174,718,992  174,718,992           0.0  cudaMemcpyToSymbol    
      0.3      171,631,313      2,027       84,672.6       76,068.0        4,080    2,602,170     136,114.8  cudaFree              
      0.2       82,969,974     12,000        6,914.2        6,088.5        3,821      313,898       4,688.8  cudaMemsetAsync       
      0.1       54,032,122      4,003       13,497.9       10,131.0        3,307      381,319      11,735.8  cudaMemset            
      0.1       41,868,033      2,027       20,655.2       18,396.0        3,193      214,659      12,110.7  cudaMalloc            
      0.0            1,404          1        1,404.0        1,404.0        1,404        1,404           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------------------------------------------------------------------------------------
     23.8   12,047,176,456      2,000    6,023,588.2    5,983,810.0    5,426,109  107,902,623  2,281,456.6  counting_sort_scatter_kernel(Particles, Particles, int *, int *, int)                               
     18.7    9,452,598,162      2,000    4,726,299.1    4,734,267.5    3,738,215    4,907,639     96,051.0  scatter_and_key_kernel(Particles, Particles, int *, int *, int *, int *, int)                       
     17.0    8,589,150,993     12,001      715,702.9      718,330.0      664,315    6,772,500     57,239.8  generate_particles_in_rect_kernel(Particles, int, int, float, float, float, float, float, float, in…
     16.5    8,357,774,082      2,000    4,178,887.0    4,214,125.0    3,042,892    4,348,605    167,791.0  ntcs_work_queue_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *, int *, i…
      7.9    4,016,009,796      2,000    2,008,004.9    2,013,840.0    1,804,309    2,027,311     26,615.9  move_particles_kernel(Particles, int, curandStateXORWOW *)                                          
      5.5    2,770,203,687        100   27,702,036.9   27,697,055.5   27,192,291   28,298,558    215,669.7  accumulate_sampling_kernel(int *, int *, Cell *, Particles)                                         
      4.8    2,446,133,137      2,000    1,223,066.6    1,224,982.0    1,166,680    1,232,342     10,333.7  mark_valid_kernel(Particles, int *, int)                                                            
      2.7    1,390,919,481      4,000      347,729.9      333,485.5        6,272      700,474    340,897.9  void cub::CUB_200700_750_NS::DeviceScanKernel<cub::CUB_200700_750_NS::DeviceScanPolicy<int, cub::CU…
      1.4      733,402,506          1  733,402,506.0  733,402,506.0  733,402,506  733,402,506          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      1.1      545,222,558      2,000      272,611.3      272,558.0      163,423      334,781     15,781.8  hss_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)               
      0.4      190,841,268      8,000       23,855.2       23,936.0        9,504       39,199      8,606.8  void cub::CUB_200700_750_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_750_NS::DeviceRadixSortP…
      0.1       36,315,076      2,000       18,157.5       18,111.5       15,136       20,096        824.4  void cub::CUB_200700_750_NS::DeviceRadixSortHistogramKernel<cub::CUB_200700_750_NS::DeviceRadixSort…
      0.0        7,251,203      4,000        1,812.8        1,824.0        1,536        2,112         85.6  void cub::CUB_200700_750_NS::DeviceScanInitKernel<cub::CUB_200700_750_NS::ScanTileState<int, (bool)…
      0.0        3,938,087      2,000        1,969.0        1,952.0        1,632        2,208         89.1  void cub::CUB_200700_750_NS::detail::for_each::static_kernel<cub::CUB_200700_750_NS::detail::for_ea…
      0.0        3,646,140      2,000        1,823.1        1,824.0        1,536        2,016         80.0  void cub::CUB_200700_750_NS::DeviceRadixSortExclusiveSumKernel<cub::CUB_200700_750_NS::DeviceRadixS…
      0.0        2,892,045          1    2,892,045.0    2,892,045.0    2,892,045    2,892,045          0.0  filter_particles_inside_ball(Particles, Particles, int, int *)                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  ------  --------  --------  --------  ----------  -----------  ------------------
     85.2      210,970,776   4,009  52,624.3   1,312.0     1,184  34,332,581  1,311,830.1  [CUDA memcpy DtoH]
      9.6       23,851,512  16,003   1,490.4   1,568.0       384      17,600        559.1  [CUDA memset]     
      5.2       12,939,098   4,000   3,234.8   3,232.0     2,656       3,744        151.9  [CUDA memcpy DtoD]
      0.0              704       1     704.0     704.0       704         704          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  ------  --------  --------  --------  --------  -----------  ------------------
  2,000.000   4,000     0.500     0.500     0.500     0.500        0.000  [CUDA memcpy DtoD]
  1,111.536  16,003     0.069     0.011     0.000     5.000        0.167  [CUDA memset]     
    322.511   4,009     0.080     0.000     0.000    52.916        2.047  [CUDA memcpy DtoH]
      0.000       1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]

Generated:
    /content/DSMC/cuda/build/report1.nsys-rep
    /content/DSMC/cuda/build/report1.sqlite
```


## Pull Request 25

https://github.com/stasostyk/DSMC/pull/25

Quick comparison for `50x50x50` grid case (comparison on the cluster):
* our previous code: 17,5seconds and taking 3573MiB
* two-GPUs: 17,5seconds (per GPU) and taking 214MiB (per GPU)

It didn't obtain speedup (due to communication overhead), but this at least allows to distribute the simulation to multiple GPUs, meaning that now bigger problem size can be solved (single GPU is limited by its memory, multiple GPUs allow to use more).

## Pull Request 28

https://github.com/stasostyk/DSMC/pull/28

Improvements for QoL. 
- Refactored collisions to reuse logic
- Launch HSS using precomputed work queue, saves lots of overhead
- Found bug where NTC was using Variable Sphere Model but HSS Hard Sphere Model. Moved both to use VHS (this fixed collision inconsistencies! :D)
- Refactored "ball" to "sphere"
- Improved Slurm scripts for cluster deployment
- Improved initial README
