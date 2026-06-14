# Used private, shared, and constant memory for better memory management

## Pull Request 9

https://github.com/stasostyk/DSMC/pull/9

Reduced `cudaMemcpy` that is done between host and device by initializing and storing particle and cell data in device (GPU), and only copying it to host when needed for printing statistics and saving final results.

Also, config data moved to device memory as well, so that it wouldn't be copied everytime it is needed.

### Improvement for CUDA BALL case

|   |   BEFORE | AFTER   |   improvement |
|---|---|---|---|
| Simulation loop  | 219792.0004 ms    |  8802.1847 ms |   ~**25x speedup** |
| memcpy HtoD, time sum |  109,813,600,788 ns  | 1,184  ns   |  HtoD transfer almost removed    |
| memcpy DtoH, time sum  | 89,497,574,390 ns  |  78,512,156 ns  |  ~1140x speedup |
| memcpy DtoD, time sum | - | 166,255,833 ns | - | 
| memcpy HtoD, mem size sum | 515,247.856 MB | 0.000 MB | HtoD transfer almost removed |
| memcpy DtoH, mem size sum | 487,662.778 MB | 264.456 MB | ~1850x less memory  |
| memcpy DtoD, mem size sum | - | 20,420.385 | - |

Note: a lot of DtoH and HtoD transfers were saved by basically only managing the particle data by GPU, but that means that some parts of the algorithm need to be creating new particles which means a lot of DtoD transfers are now happening. (That's an idea for another improvement.)

HtoD transfer almost removed, it is only used one: to initialize config struct in the device memory.

### Stats
The statistics (time and profiling) can be found in this branche's notebook: 
https://github.com/stasostyk/DSMC/blob/cuda_opt1_memcpy/cuda/colab_run.ipynb
but I will paste it here as well:
- params:
  - NX=NY=NZ=10
  - particlesPerCellTarget = 200
  - nSteps = 2000
- SERIAL BALL case
  - INITIALIZATION. Elapsed time: 71.8265 ms
  - SIMULATION LOOP. Elapsed time: 120840.2735 ms
  - ALL PROGRAM FINISHED. Elapsed time: 120922.8004 ms
- SERIAL WING case
  - INITIALIZATION. Elapsed time: 81.9364 ms
  - SIMULATION LOOP. Elapsed time: 113330.7752 ms
  - ALL PROGRAM FINISHED. Elapsed time: 113428.9176 ms
- CUDA BALL case
  - INITIALIZATION. Elapsed time: 274.1557 ms   (prev: 342.0782 ms)
  - SIMULATION LOOP. Elapsed time: 8802.1847 ms  (prev: 219792.0004 ms)
  - ALL PROGRAM FINISHED. Elapsed time: 9102.1725 ms  (prev: 220147.3259 ms)
- CUDA WING case
  - INITIALIZATION. Elapsed time: 286.3415 ms   (prev: 205.3593 ms)
  - SIMULATION LOOP. Elapsed time: 8463.4146 ms  (prev: 218552.1742 ms)
  - ALL PROGRAM FINISHED. Elapsed time: 8767.5032 ms  (prev: 218777.8221 ms)


### NSYS profiling info for CUDA BALL case

```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ----------------------
     94.4    9,512,352,647        103  92,352,938.3  100,141,961.0   297,817  385,269,552  41,309,232.6  poll                  
      5.2      522,419,803     14,547      35,912.5       23,749.0       318   21,431,411     215,351.5  ioctl                 
      0.3       27,077,312      4,007       6,757.5        5,656.0     3,186    2,094,342      34,195.3  munmap                
      0.1       13,454,318      2,016       6,673.8        6,171.0     2,486      311,851       7,705.6  mmap                  
      0.0        1,922,743         29      66,301.5       11,139.0     8,848    1,289,787     236,725.1  mmap64                
      0.0          578,544          8      72,318.0       75,908.0    44,914      111,934      20,625.5  sem_timedwait         
      0.0          513,451          1     513,451.0      513,451.0   513,451      513,451           0.0  pthread_cond_wait     
      0.0          428,857         47       9,124.6        7,931.0     1,879       23,820       4,594.3  open64                
      0.0          407,621         39      10,451.8        3,818.0     1,658       88,493      17,047.6  fopen                 
      0.0          224,028         32       7,000.9        1,528.5       747       69,442      16,384.0  fclose                
      0.0          202,773         12      16,897.8       20,530.5     1,967       36,779      11,256.2  write                 
      0.0          139,110          2      69,555.0       69,555.0    61,776       77,334      11,001.2  pthread_create        
      0.0           87,417         20       4,370.9           56.5        45       84,471      18,857.4  fgets                 
      0.0           35,790         62         577.3          569.5       168        1,266         230.0  fcntl                 
      0.0           28,776          6       4,796.0        5,682.5     1,923        6,855       2,030.5  open                  
      0.0           28,483         31         918.8           90.0        43       13,177       2,508.0  fwrite                
      0.0           24,469         15       1,631.3        1,564.0       793        3,610         757.8  read                  
      0.0           24,172          2      12,086.0       12,086.0     8,506       15,666       5,062.9  socket                
      0.0           17,062          3       5,687.3        5,786.0     2,856        8,420       2,783.3  pipe2                 
      0.0            9,835          1       9,835.0        9,835.0     9,835        9,835           0.0  connect               
      0.0            6,749          2       3,374.5        3,374.5     2,125        4,624       1,767.1  pthread_cond_broadcast
      0.0            5,231        110          47.6           36.0        35          197          23.6  fputc                 
      0.0            3,083          8         385.4          366.5       316          541          70.1  dup                   
      0.0            1,645          1       1,645.0        1,645.0     1,645        1,645           0.0  bind                  
      0.0            1,046          1       1,046.0        1,046.0     1,046        1,046           0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -----------  ---------  --------  -----------  -----------  ----------------------
     87.4    7,918,736,163      6,023  1,314,749.5  239,403.0     9,642   20,272,105  1,674,360.4  cudaMemcpy            
      5.7      517,641,259      6,006     86,187.4    9,110.5     3,653  176,042,855  2,272,501.4  cudaMalloc            
      4.5      405,090,563      6,006     67,447.6   83,392.5     5,143    1,764,591     56,582.6  cudaFree              
      2.1      189,973,931     22,102      8,595.3    6,932.0     3,696    2,663,968     25,777.7  cudaLaunchKernel      
      0.4       32,670,778      4,001      8,165.7    7,467.0     4,113       63,812      3,959.5  cudaMemset            
      0.0            1,228          1      1,228.0    1,228.0     1,228        1,228          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     85.8    7,019,427,832      2,000   3,509,713.9   3,523,909.0   2,499,135   3,825,469    129,312.5  no_time_counter_scheme_kernel(int *, Particle *, int *, int *, double, double, Config *, curandStat…
      3.9      316,246,336      2,000     158,123.2     158,300.0     148,956     161,884      1,722.3  filter_particles_out_of_bounds(Config *, Particle *, Particle *, int, int *)                        
      3.9      315,767,923     12,001      26,311.8      26,048.0      22,239     297,624      3,101.0  generate_particles_in_rect_kernel(Particle *, Config *, int, int, double, double, double, double, d…
      3.1      253,285,136      2,000     126,642.6     124,989.0     119,389     235,674     10,950.0  move_particles_kernel(Particle *, Config *, int, curandStateXORWOW *)                               
      2.0      167,070,052      2,000      83,535.0      83,389.0      71,742     103,101      2,222.7  bin_particles_kernel(Particle *, int *, int *, int, double, double, double)                         
      1.2       96,468,894        100     964,688.9     964,743.0     946,728     982,343      7,492.1  accumulate_sampling_kernel(int *, int *, Cell *, Particle *)                                        
      0.2       13,132,684          1  13,132,684.0  13,132,684.0  13,132,684  13,132,684          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.0        3,273,841      2,000       1,636.9       1,600.0       1,503       2,624        126.5  reset_cell_count_kernel(int *)                                                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  ----------  -----------  ------------------
     67.5      166,255,833  2,000  83,127.9  83,294.0    73,950      92,094      1,251.8  [CUDA memcpy DtoD]
     31.9       78,512,156  4,022  19,520.7   1,184.0     1,120  18,751,610    406,076.4  [CUDA memcpy DtoH]
      0.7        1,669,012  4,001     417.1     384.0       351       2,848        172.4  [CUDA memset]     
      0.0            1,184      1   1,184.0   1,184.0     1,184       1,184          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
 20,420.385  2,000    10.210    10.235     9.593    10.309        0.113  [CUDA memcpy DtoD]
    264.456  4,022     0.066     0.000     0.000    24.000        1.254  [CUDA memcpy DtoH]
      0.056  4,001     0.000     0.000     0.000     0.040        0.001  [CUDA memset]     
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]
```


## Pull Request 10

https://github.com/stasostyk/DSMC/pull/10

### Changes

After the last optimizations, the bottleneck seems to be `no_time_counter_scheme_kernel` which took most of the kernel execution time. To optimize it, several inefficiencies have been spotted, mainly inefficient computation and some unnecessary global memory access. To improve, the following changes have been done (to this kernel, but also to other as well):

* Config is now also stored in device's constant memory, so any required value can be accessed fast from GPU as well (this takes advantage of read only cache).
* A lot of values that are derived from just config's values are now also saved in the config's struct so that they could be easily accessed and be already precomputed when device needs it. This also includes some precomputed multipliers to avoid always computing the same values in device.
* The `pow()` function was changed into `sqrt()` with the assumption of `omega=0.75`.
* curandStates are now firstly taken into a local variable (privatization) to avoid always fetching and updating them in the global memory.   
* Removed `while(i==j)` when searching for two random indices, now done mathematically (reduces call to random and reduces warp divergence).
* Reducing `atomic_add` calls by firstly storing the counts in the shared memory, then doing reduction on shared memory, and only then one thread per block calls `atomic_add`.
* Changed kernels' block sizes (potentially reduces register spilling).
* Removed calls of global diagnostics during simulation loop, as they require DeviceToHost memory transfer of sample data. This can still be done if needed for debugging, but for benchmarking the performance it is not needed.


### Improvement

For CUDA BALL case, grid NX=NY=NZ=10 and particlesPerCellTarget=200, as in the previous PR.

|   |   BEFORE | AFTER   |   improvement |
|---|---|---|---|
| Simulation loop  |  8802.1847 ms   | 3587.6836 ms  | ~**2.5x speedup** |
| `no_time_counter_scheme_kernel` total time sum | 7,019,427,832 ns |  1,724,539,825 | ~**4x speedup** |

### Stats for CUDA BALL case (10x10x10 grid, particlesPerCellTarget=200)

Main code:
```
Running Ball case.
INITIALIZATION. Elapsed time: 251.4371 ms
SIMULATION LOOP. Elapsed time: 3587.6836 ms
step=2000
  NP=213270
  mean_u=(9.633724e+02, -5.255957e+00, -3.535287e-01)
  T=3.547693e+02
  totalCollisions = 20440955
ALL PROGRAM FINISHED. Elapsed time: 3868.0277 ms
```

NSYS profiling:
```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ----------------------
     89.0    4,167,509,946         50  83,350,198.9  100,147,233.5   327,557  358,225,495  55,485,304.6  poll                  
     10.3      480,101,754     14,547      33,003.5       21,937.0       347   17,550,421     192,904.6  ioctl                 
      0.4       19,389,250      4,007       4,838.8        4,650.0     2,769       48,776       2,299.7  munmap                
      0.2       10,063,111      2,016       4,991.6        4,677.5     2,337       97,307       2,624.4  mmap                  
      0.0        1,809,054         29      62,381.2       11,648.0     7,708    1,236,921     226,606.3  mmap64                
      0.0          546,623          1     546,623.0      546,623.0   546,623      546,623           0.0  pthread_cond_wait     
      0.0          459,434         10      45,943.4       42,871.0    36,299       66,473       9,623.1  sem_timedwait         
      0.0          402,556         39      10,321.9        3,919.0     1,413       95,225      17,761.0  fopen                 
      0.0          381,228         47       8,111.2        7,677.0     1,995       15,965       3,289.9  open64                
      0.0          213,387         32       6,668.3        1,353.5       758       68,432      16,064.5  fclose                
      0.0          175,864         12      14,655.3        6,088.0     2,306       58,226      16,132.5  write                 
      0.0          146,527          2      73,263.5       73,263.5    65,127       81,400      11,506.7  pthread_create        
      0.0           36,662         20       1,833.1           45.0        44       35,712       7,974.3  fgets                 
      0.0           30,644         62         494.3          515.5       172          915         167.3  fcntl                 
      0.0           28,939         31         933.5           77.0        47       13,868       2,628.0  fwrite                
      0.0           26,501          6       4,416.8        4,561.5     1,581        6,628       1,994.5  open                  
      0.0           22,942         15       1,529.5        1,261.0       875        2,996         643.6  read                  
      0.0           21,575          3       7,191.7        6,037.0     5,865        9,673       2,150.6  pipe2                 
      0.0           21,534          2      10,767.0       10,767.0     7,277       14,257       4,935.6  socket                
      0.0           13,701          1      13,701.0       13,701.0    13,701       13,701           0.0  connect               
      0.0            7,897          2       3,948.5        3,948.5     2,334        5,563       2,283.2  pthread_cond_broadcast
      0.0            5,374        110          48.9           40.0        35          461          42.3  fputc                 
      0.0            2,925          8         365.6          393.0       241          429          69.8  dup                   
      0.0            1,484          1       1,484.0        1,484.0     1,484        1,484           0.0  bind                  
      0.0              800          1         800.0          800.0       800          800           0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------
     74.3    2,815,532,187      6,002      469,099.0      416,766.0        8,921   16,571,916    474,239.7  cudaMemcpy            
      9.5      361,183,423      6,005       60,147.1       80,074.0        4,606      618,219     41,200.2  cudaFree              
      7.1      268,794,365      6,005       44,761.8        6,098.0        3,482    1,016,848     57,611.2  cudaMalloc            
      4.6      175,455,773          1  175,455,773.0  175,455,773.0  175,455,773  175,455,773          0.0  cudaMemcpyToSymbol    
      3.7      140,468,108     22,102        6,355.4        5,152.0        3,460      184,005      3,943.3  cudaLaunchKernel      
      0.7       28,185,041      4,001        7,044.5        6,464.0        4,508       64,371      3,397.4  cudaMemset            
      0.0            2,243          1        2,243.0        2,243.0        2,243        2,243          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     55.7    1,724,539,825      2,000     862,269.9     854,124.5     309,496   1,409,920     80,526.3  no_time_counter_scheme_kernel(int *, Particle *, int *, int *, curandStateXORWOW *)                 
     13.9      430,647,094      2,000     215,323.5     215,163.0     205,147     247,706      3,058.8  move_particles_kernel(Particle *, int, curandStateXORWOW *)                                         
     13.2      409,953,953     12,001      34,160.0      35,391.0      24,032     529,300      8,789.6  generate_particles_in_rect_kernel(Particle *, int, int, double, double, double, double, double, dou…
     10.1      312,854,042      2,000     156,427.0     156,700.0     146,653     160,829      1,936.8  filter_particles_out_of_bounds(Particle *, Particle *, int, int *)                                  
      5.4      168,476,500      2,000      84,238.3      83,550.0      76,478     113,150      4,789.9  bin_particles_kernel(Particle *, int *, int *, int)                                                 
      1.2       35,661,653        100     356,616.5     355,703.5     341,337     379,095      8,271.3  accumulate_sampling_kernel(int *, int *, Cell *, Particle *)                                        
      0.4       13,583,561          1  13,583,561.0  13,583,561.0  13,583,561  13,583,561          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.1        3,015,548      2,000       1,507.8       1,472.0       1,344       2,528        184.4  reset_cell_count_kernel(int *)                                                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  ----------  -----------  ------------------
     88.3      166,427,689  2,000  83,213.8  83,390.0    76,510      94,878      1,324.4  [CUDA memcpy DtoD]
     10.8       20,270,533  4,002   5,065.1   1,216.0       512  15,306,626    241,938.9  [CUDA memcpy DtoH]
      0.9        1,688,097  4,001     421.9     384.0       351       3,136        180.6  [CUDA memset]     
      0.0              672      1     672.0     672.0       672         672          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
 20,415.110  2,000    10.208    10.233     9.595    10.287        0.113  [CUDA memcpy DtoD]
     24.056  4,002     0.006     0.000     0.000    24.000        0.379  [CUDA memcpy DtoH]
      0.056  4,001     0.000     0.000     0.000     0.040        0.001  [CUDA memset]     
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]
```

### Stats for CUDA BALL case (50x50x50 grid, particlesPerCellTarget=200)

Main code:
```
INITIALIZATION. Elapsed time: 211.5471 ms
SIMULATION LOOP. Elapsed time: 280032.9743 ms
step=2000
  NP=26479879
  mean_u=(9.756956e+02, -4.032339e+00, -3.742569e-02)
  T=3.382912e+02
  totalCollisions = 2554039239
ALL PROGRAM FINISHED. Elapsed time: 284878.6337 ms
```

NSYS profiling:
```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  -----------  ----------------------
     98.3  287,021,012,073      2,872  99,937,678.3  100,138,153.0     1,462  381,221,909  7,871,423.6  poll                  
      1.7    5,039,099,013     14,556     346,187.1       68,603.5       320   17,736,447    595,753.3  ioctl                 
      0.0       32,337,544      4,010       8,064.2        7,219.5     1,689       49,746      2,932.7  munmap                
      0.0       15,683,548      2,017       7,775.7        6,692.0     2,766      775,984     17,328.3  mmap                  
      0.0        8,710,323         32     272,197.6        1,305.0       725    7,574,741  1,345,229.4  fclose                
      0.0        3,366,068         39      86,309.4        3,757.0     1,430    1,613,583    343,905.1  fopen                 
      0.0        1,826,245         29      62,974.0       10,394.0     8,446    1,258,844    230,995.4  mmap64                
      0.0          571,656         10      57,165.6       57,211.5    30,202       95,802     17,061.8  sem_timedwait         
      0.0          507,469          1     507,469.0      507,469.0   507,469      507,469          0.0  pthread_cond_wait     
      0.0          382,021         47       8,128.1        7,955.0     2,103       19,605      3,032.5  open64                
      0.0          191,911         12      15,992.6       18,245.5     2,067       28,126      9,266.8  write                 
      0.0          179,052          2      89,526.0       89,526.0    61,417      117,635     39,752.1  pthread_create        
      0.0          168,297      2,550          66.0           59.0        35          662         34.0  fputc                 
      0.0           49,370         31       1,592.6          182.0        42       25,961      4,878.3  fwrite                
      0.0           47,824         20       2,391.2           45.0        45       46,920     10,481.0  fgets                 
      0.0           34,688         62         559.5          541.5       168        3,852        455.4  fcntl                 
      0.0           29,437          6       4,906.2        4,931.5     1,620        8,005      2,435.3  open                  
      0.0           23,806         15       1,587.1        1,332.0       904        3,542        720.3  read                  
      0.0           21,098          2      10,549.0       10,549.0     6,805       14,293      5,294.8  socket                
      0.0           17,977          1      17,977.0       17,977.0    17,977       17,977          0.0  connect               
      0.0           15,448          3       5,149.3        6,222.0     2,736        6,490      2,094.3  pipe2                 
      0.0            6,940          2       3,470.0        3,470.0     2,108        4,832      1,926.2  pthread_cond_broadcast
      0.0            3,045          8         380.6          380.0       272          452         56.7  dup                   
      0.0            1,460          1       1,460.0        1,460.0     1,460        1,460          0.0  bind                  
      0.0              950          1         950.0          950.0       950          950          0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)      Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -------------  ------------  ----------------------
     90.6  258,467,207,525      6,002   43,063,513.4   54,935,017.0       23,062  3,351,617,853  62,819,619.8  cudaMemcpy            
      8.6   24,660,159,781      6,005    4,106,604.5    1,447,575.0        9,547     11,960,444   4,653,903.7  cudaFree              
      0.7    1,859,084,210      6,005      309,589.4       12,766.0        6,278      4,685,036     464,573.7  cudaMalloc            
      0.1      230,622,741     22,102       10,434.5        7,005.5        3,483      2,092,260      18,786.6  cudaLaunchKernel      
      0.1      168,351,996          1  168,351,996.0  168,351,996.0  168,351,996    168,351,996           0.0  cudaMemcpyToSymbol    
      0.0       46,050,632      4,001       11,509.8       10,878.0        5,958         71,648       5,130.8  cudaMemset            
      0.0            1,103          1        1,103.0        1,103.0        1,103          1,103           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances     Avg (ns)         Med (ns)        Min (ns)       Max (ns)     StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------------  ---------------  -------------  -------------  -----------  ----------------------------------------------------------------------------------------------------
     22.0   56,574,485,326      2,000     28,287,242.7     28,146,560.5     27,529,192     41,960,397  1,036,655.8  bin_particles_kernel(Particle *, int *, int *, int)                                                 
     21.3   54,734,401,778      2,000     27,367,200.9     27,477,099.5     25,231,425     28,429,603    623,865.5  move_particles_kernel(Particle *, int, curandStateXORWOW *)                                         
     20.8   53,493,838,121      2,000     26,746,919.1     26,754,892.5     18,190,922     30,403,543  1,010,690.2  no_time_counter_scheme_kernel(int *, Particle *, int *, int *, curandStateXORWOW *)                 
     18.5   47,507,302,110     12,001      3,958,612.0      4,030,772.0      2,495,071     43,611,974  1,067,924.9  generate_particles_in_rect_kernel(Particle *, int, int, double, double, double, double, double, dou…
     14.3   36,816,833,796      2,000     18,408,416.9     18,429,864.0     17,579,593     18,553,382    133,904.0  filter_particles_out_of_bounds(Particle *, Particle *, int, int *)                                  
      2.0    5,265,191,529        100     52,651,915.3     52,659,362.0     52,111,209     53,017,229    201,582.5  accumulate_sampling_kernel(int *, int *, Cell *, Particle *)                                        
      1.0    2,553,585,789          1  2,553,585,789.0  2,553,585,789.0  2,553,585,789  2,553,585,789          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.0       17,393,670      2,000          8,696.8          8,608.0          7,039         10,431        610.0  reset_cell_count_kernel(int *)                                                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)     Max (ns)     StdDev (ns)       Operation     
 --------  ---------------  -----  ------------  ------------  ----------  -------------  ------------  ------------------
     86.3   21,180,340,989  2,000  10,590,170.5  10,603,306.0  10,084,868     10,717,404      82,407.9  [CUDA memcpy DtoD]
     13.7    3,361,870,210  4,002     840,047.5       2,240.0       1,887  3,350,520,509  52,963,110.9  [CUDA memcpy DtoH]
      0.0        3,923,396  4,001         980.6         928.0         767         17,377         357.2  [CUDA memset]     
      0.0              704      1         704.0         704.0         704            704           0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

  Total (MB)    Count  Avg (MB)   Med (MB)   Min (MB)   Max (MB)   StdDev (MB)      Operation     
 -------------  -----  ---------  ---------  ---------  ---------  -----------  ------------------
 2,536,255.830  2,000  1,268.128  1,271.138  1,198.991  1,271.805       12.095  [CUDA memcpy DtoD]
     4,805.016  4,002      1.201      0.000      0.000  4,800.000       75.876  [CUDA memcpy DtoH]
         5.016  4,001      0.001      0.000      0.000      5.000        0.079  [CUDA memset]     
         0.000      1      0.000      0.000      0.000      0.000        0.000  [CUDA memcpy HtoD]
```


## Pull Request 11

https://github.com/stasostyk/DSMC/pull/11

### Changes

After the last changes it can still be seen that there is a lot of DtoD traffic, which is generated when new particles are being created, and some DtoH traffic which is not huge but potentially has a slight stall on the pipeline (memcpy is a blocking operation). That's why the following optimizations were made:
* Removed DtoD traffic by introducing duplicated array for newly generated particles. The pointers are then simply switched with the old array. This way there is no overhead of always creating a new structure. However, the downside is that more memory is needed (before 50x50x50 case required ~11GB of GPU RAM, now it requires ~15GB of GPU RAM).
* When copying particles into host memory (before printing or saving results), now only the actual amount of particles, i.e. `NP`, is copied instead of the whole array of length `MAX_PARTICLES` which potentially has a lot of unused particles.
* Total collision count is stored in GPU, which helps to avoid some small blocking memcpy when generating collisions. This is a small change, still this alone resulted in a small, but non negligible, ~3-4% speedup of the simulation loop.

### Improvement

For CUDA BALL case, grid NX=NY=NZ=50 and particlesPerCellTarget=200.

|   |   BEFORE | AFTER   |   improvement |
|---|---|---|---|
| Simulation loop  |  280032.9743 ms   | 235544.5583 ms  | ~**18.9% speedup** |
| DtoD memcpy total time sum | 21,180,340,989 ns |  - | DtoD traffic removed |
| DtoH memcpy total time sum | 3,361,870,210 ns | 896,284,925 ns | ~**3.75x speedup** |

### Stats for CUDA Ball case, NX=NY=NZ=50, cell particle target = 200

Main code:
```
Running Ball case.
INITIALIZATION. Elapsed time: 282.5181 ms
SIMULATION LOOP. Elapsed time: 235544.5583 ms
step=2000
  NP=26483217
  mean_u=(9.757264e+02, -4.037984e+00, -3.046750e-02)
  T=3.382115e+02
  totalCollisions = 2554227274
ALL PROGRAM FINISHED. Elapsed time: 238021.5174 ms
```

NSYS profiling:
```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  -----------  ----------------------
     99.9  245,982,454,739      2,463  99,871,073.8  100,146,845.0     1,673  315,489,300  7,681,458.8  poll                  
      0.1      181,544,485        563     322,459.1       16,315.0       354   18,496,609    996,732.3  ioctl                 
      0.0        5,711,247         32     178,476.5        1,455.5       854    4,550,404    819,521.9  fclose                
      0.0        3,386,034         39      86,821.4        3,753.0     1,515    1,702,246    344,884.9  fopen                 
      0.0        1,851,008         29      63,827.9       11,051.0     7,863    1,260,752    230,989.4  mmap64                
      0.0          683,760         10      68,376.0       71,583.5    40,412       98,479     18,361.9  sem_timedwait         
      0.0          399,396         47       8,497.8        7,956.0     2,668       23,687      3,741.4  open64                
      0.0          300,199         12      25,016.6       29,838.0     1,349       43,625     12,312.4  write                 
      0.0          204,071         18      11,337.3        8,655.5     2,920       60,460     13,202.3  mmap                  
      0.0          169,698          2      84,849.0       84,849.0    69,114      100,584     22,252.7  pthread_create        
      0.0          169,175      2,550          66.3           57.0        35        3,553        103.0  fputc                 
      0.0          115,401          1     115,401.0      115,401.0   115,401      115,401          0.0  pthread_cond_wait     
      0.0          106,039         12       8,836.6        6,257.5     2,463       34,226      8,452.0  munmap                
      0.0           44,048         31       1,420.9          171.0        43       16,378      3,433.5  fwrite                
      0.0           38,185         20       1,909.3           48.0        47       37,204      8,307.5  fgets                 
      0.0           34,599          2      17,299.5       17,299.5     2,462       32,137     20,983.4  pthread_cond_broadcast
      0.0           33,029         62         532.7          554.0       187        1,266        201.7  fcntl                 
      0.0           30,109          6       5,018.2        5,391.5     1,725        7,419      1,943.1  open                  
      0.0           26,551         15       1,770.1        1,532.0     1,262        3,132        552.1  read                  
      0.0           23,279          2      11,639.5       11,639.5     6,097       17,182      7,838.3  socket                
      0.0           15,767          3       5,255.7        6,388.0     2,522        6,857      2,379.0  pipe2                 
      0.0           10,237          1      10,237.0       10,237.0    10,237       10,237          0.0  connect               
      0.0            3,139          8         392.4          413.5       297          459         60.2  dup                   
      0.0            1,713          1       1,713.0        1,713.0     1,713        1,713          0.0  bind                  
      0.0              897          1         897.0          897.0       897          897          0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)      Max (ns)     StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -------------  ------------  ----------------------
     99.8  244,044,345,118      2,003  121,839,413.4  122,854,331.0    3,732,840  2,951,171,909  66,724,901.7  cudaMemcpy            
      0.1      216,795,592     22,102        9,808.9        6,413.5        3,500      2,095,340      17,614.2  cudaLaunchKernel      
      0.1      178,381,783          1  178,381,783.0  178,381,783.0  178,381,783    178,381,783           0.0  cudaMemcpyToSymbol    
      0.0       72,832,682      2,007       36,289.3       26,686.0        9,012      4,394,511     163,666.7  cudaFree              
      0.0       33,362,710      2,007       16,623.2       12,887.0        8,779        489,549      21,380.3  cudaMalloc            
      0.0       24,992,721      2,002       12,483.9       11,280.5        8,033         67,703       3,533.6  cudaMemset            
      0.0            1,865          1        1,865.0        1,865.0        1,865          1,865           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances     Avg (ns)         Med (ns)        Min (ns)       Max (ns)     StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------------  ---------------  -------------  -------------  -----------  ----------------------------------------------------------------------------------------------------
     21.5   52,292,863,610      2,000     26,146,431.8     26,184,825.5     24,635,475     26,554,735    265,560.5  move_particles_kernel(Particle *, int, curandStateXORWOW *)                                         
     20.4   49,523,327,850     12,001      4,126,600.1      4,281,444.0      2,673,127     47,368,813  1,124,406.0  generate_particles_in_rect_kernel(Particle *, int, int, double, double, double, double, double, dou…
     19.9   48,318,685,846      2,000     24,159,342.9     23,963,905.0     22,954,456     37,043,913  1,071,363.0  bin_particles_kernel(Particle *, int *, int *, int)                                                 
     19.8   48,265,338,022      2,000     24,132,669.0     24,092,312.5     16,000,459     29,071,007  1,104,619.8  no_time_counter_scheme_kernel(unsigned long long *, Particle *, int *, int *, curandStateXORWOW *)  
     15.1   36,783,053,472      2,000     18,391,526.7     18,416,586.0     17,486,988     18,499,474    151,893.6  filter_particles_out_of_bounds(Particle *, Particle *, int, int *)                                  
      2.1    5,217,935,081        100     52,179,350.8     52,221,026.5     51,594,913     52,657,576    180,369.4  accumulate_sampling_kernel(int *, int *, Cell *, Particle *)                                        
      1.2    2,832,550,205          1  2,832,550,205.0  2,832,550,205.0  2,832,550,205  2,832,550,205          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.0       15,292,392      2,000          7,646.2          7,600.0          6,720          8,608        452.7  reset_cell_count_kernel(int *)                                                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)  Min (ns)   Max (ns)    StdDev (ns)       Operation     
 --------  ---------------  -----  ---------  --------  --------  -----------  ------------  ------------------
     99.9      896,284,925  2,003  447,471.3   1,728.0       736  890,275,171  19,892,252.5  [CUDA memcpy DtoH]
      0.1        1,088,167  2,002      543.5     512.0       479       18,719         411.6  [CUDA memset]     
      0.0              704      1      704.0     704.0       704          704           0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)   StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  ---------  -----------  ------------------
  1,275.989  2,003     0.637     0.000     0.000  1,270.981       28.399  [CUDA memcpy DtoH]
      5.008  2,002     0.003     0.000     0.000      5.000        0.112  [CUDA memset]     
      0.000      1     0.000     0.000     0.000      0.000        0.000  [CUDA memcpy HtoD]
```
