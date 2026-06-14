# Sorted, ordered particles

## Pull Request 22

https://github.com/stasostyk/DSMC/pull/22

### Main optimization
* the workflow changed. Previously: move particles -> generate new particles -> filter out of bounds -> reindex to cells (binning). Now: move particles -> generate new particles -> mark valid (inside bounds) -> compute particle IDs (scan of valid) -> scatter to new positions (i.e. reorder particles) and calculate cellKeys and cellCounts -> scan cellCoutns (to get cellCountPrefixSums, i.e. to be able to get the particles easily for each cell). **This way it adds more overhead in this workflow, but then collision kernel is faster due to more coalescent accesses** 
* `cellList` removed, and now the particles are stored in order, i.e. particles with IDs `[cellCountPrefixSum[cell], cellCountPrefixSum[cell] + cellCount[cell])` belong to the `cell`.
* collision scheme goes over cells in a row major order, i.e. the fastest varying directions of gpu grid and cell array match.

### Other changes to optimize the code, that not neccesarily gave a significant speedup
* Making the grid size `Lx, Ly, Lz` and `DL` as floats instead of doubles. Same for stream variables `MaFree, PFree, TFree`, cell dimensions `cellVolumen, dx, dy, dz`. All these variables do not need high precision.
* Reducing `MAX_PARTICLES` and `MAX_PARTICLES_PER_CELL` to have tighter arrays that still can fit the particles. (this is essentially just scaling the configuration of the problem, so not a speedup on its own) .
* When calculating probability of collision, I take the square of it in order to avoid calculating square root. For this a new value is pre-computed in advance, namely `ntcs_collisionProbMultiplierSquared`.
* `d_new_NP` is allocated only once
* `generate_particles` kernel stores `x,y,z` locally at first to reduce calls to the global memory.

### Refactor changes
* Removed defines `WING_CASE` and `BALL_CASE`, now the case can be passed as the command line argument, which means that only one executable needs to be compiled. Also makes the code more flexible and readable. The object variables are stored in `object.h` file, and the config struct can store multiple wings and spheres if needed.
* Because of the previous change, I have also included one more case with one wing and two spheres. But many more scenarios could be created.
* Change to `plot.py` to pass the `z_val` to be displayed.


### Comparison

#### setup
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
| Simulation loop  |  85137.6991 ms   | 78016.3177 ms  | ~**9,1% speedup** |
| `no_time_counter_scheme_kernel` (total time) |  31,719,701,209 ns   | 22,325,947,517 ns  | ~**42,1% speedup** |
| `accumulate_sampling_kernel` (total time) |  4,094,276,972 ns   | 2,756,624,005 ns  | ~**48,5% speedup** |
| Filtering and indexing | Total time of `bin_particles_kernel` + `filter_particles_out_of_bounds`+`reorder_particles_by_cell_kernel`:  21,391,025,756 ns   | Total time of `counting_sort_scatter_kernel`+`scatter_and_key_kernel`+`mark_valid_kernel` +`cub::DeviceScanKernel`: 25,737,910,757 | ~16,9% slowdown (but is compensated by the speedups mentioned above) |

#### stats before
```
/content/DSMC/cuda/build
Running Ball case.
INITIALIZATION. Elapsed time: 1310.1177 ms
SIMULATION LOOP. Elapsed time: 85137.6991 ms
step=2000
  NP=13224906
  mean_u=(9.764276e+02, -4.015648e+00, 6.026279e-02)
  T=3.383248e+02
  totalCollisions = 1284013502
ALL PROGRAM FINISHED. Elapsed time: 87678.1242 ms
Generating '/tmp/nsys-report-4a9e.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/DSMC/cuda/build/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  ---------  -----------  ------------  ----------------------
     99.8   87,554,960,727        882  99,268,663.0  100,142,388.5    326,607  372,227,146  13,972,659.8  poll                  
      0.2      172,543,636        631     273,444.7       14,216.0        365   17,168,136     876,863.8  ioctl                 
      0.0        1,834,269         29      63,250.7       10,319.0      8,400    1,205,315     220,743.0  mmap64                
      0.0        1,079,952          1   1,079,952.0    1,079,952.0  1,079,952    1,079,952           0.0  pthread_cond_wait     
      0.0          688,391         10      68,839.1       51,183.5     29,567      246,869      64,091.1  sem_timedwait         
      0.0          399,316         47       8,496.1        8,127.0      2,009       17,566       3,427.1  open64                
      0.0          391,594         39      10,040.9        3,871.0      1,526       80,385      17,037.2  fopen                 
      0.0          288,199         12      24,016.6       22,182.0      2,650       71,696      18,541.8  write                 
      0.0          221,680         28       7,917.1        3,253.0      1,776       66,889      12,734.8  mmap                  
      0.0          204,349      2,550          80.1           59.0         35       12,755         301.9  fputc                 
      0.0          134,790         33       4,084.5        3,517.0      2,344        7,890       1,524.9  munmap                
      0.0          122,932          2      61,466.0       61,466.0     49,299       73,633      17,206.7  pthread_create        
      0.0           79,674         32       2,489.8        1,438.0        799       16,366       3,110.1  fclose                
      0.0           46,649         20       2,332.5           45.0         44       45,742      10,217.6  fgets                 
      0.0           37,872         15       2,524.8        1,455.0      1,188       12,976       3,031.0  read                  
      0.0           34,038         31       1,098.0          117.0         44       13,675       2,679.6  fwrite                
      0.0           33,092         62         533.7          576.5        161        1,098         199.7  fcntl                 
      0.0           28,121          6       4,686.8        4,970.5      2,564        6,543       1,496.8  open                  
      0.0           18,075          2       9,037.5        9,037.5      6,826       11,249       3,127.5  socket                
      0.0           14,767          3       4,922.3        5,911.0      2,380        6,476       2,219.8  pipe2                 
      0.0            9,377          1       9,377.0        9,377.0      9,377        9,377           0.0  connect               
      0.0            8,604          2       4,302.0        4,302.0      2,519        6,085       2,521.5  pthread_cond_broadcast
      0.0            2,709          8         338.6          345.5        243          391          44.2  dup                   
      0.0            1,554          1       1,554.0        1,554.0      1,554        1,554           0.0  bind                  
      0.0            1,414          1       1,414.0        1,414.0      1,414        1,414           0.0  listen                
      0.0              533         10          53.3           36.0         35          207          54.0  fflush                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     99.5   85,832,856,084      2,009   42,724,169.3   38,698,606.0    3,279,898  755,237,603  21,261,111.5  cudaMemcpy            
      0.2      168,843,807          1  168,843,807.0  168,843,807.0  168,843,807  168,843,807           0.0  cudaMemcpyToSymbol    
      0.2      146,709,889     22,343        6,566.3        5,505.0        3,196    1,307,128      11,701.0  cudaLaunchKernel      
      0.1       57,313,245      4,003       14,317.6       13,648.0        3,421      433,873      13,399.5  cudaMemset            
      0.1       52,651,691      2,020       26,065.2       13,982.0        4,556    2,286,856     122,205.7  cudaFree              
      0.0       24,463,381      2,020       12,110.6        9,789.5        4,304      223,743      10,300.4  cudaMalloc            
      0.0            1,979          1        1,979.0        1,979.0        1,979        1,979           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------------------------------------------------------------------------------------
     37.0   31,719,701,209      2,000   15,859,850.6   16,135,110.0   10,825,138   16,548,140    758,932.3  no_time_counter_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)   
     21.2   18,132,375,116      2,000    9,066,187.6    9,097,841.5    8,347,652    9,199,775    121,422.0  move_particles_kernel(Particles, int, curandStateXORWOW *)                                          
     11.7    9,991,419,384      2,000    4,995,709.7    4,993,568.0    3,760,517   16,341,835    520,392.4  bin_particles_kernel(Particles, int *, int *, int)                                                  
     10.3    8,868,602,935     12,001      738,988.7      741,434.0      681,753    6,845,103     57,607.9  generate_particles_in_rect_kernel(Particles, int, int, float, float, float, float, float, float, in…
      8.7    7,437,827,007      2,000    3,718,913.5    3,722,960.0    3,454,535    3,798,870     55,830.9  filter_particles_out_of_bounds(Particles, Particles, int, int *)                                    
      4.8    4,094,276,972        100   40,942,769.7   41,373,833.5   38,660,809   42,028,781  1,060,096.1  accumulate_sampling_kernel(int *, int *, Cell *, Particles)                                         
      4.6    3,961,779,365         80   49,522,242.1   49,629,020.0   46,079,221   49,904,959    596,472.9  reorder_particles_by_cell_kernel(Particles, Particles, int *, int *, int *, int)                    
      0.9      752,472,672      2,000      376,236.3      380,684.0      178,495      510,074     54,200.9  hss_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)               
      0.9      745,563,385          1  745,563,385.0  745,563,385.0  745,563,385  745,563,385          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.0        2,882,379          1    2,882,379.0    2,882,379.0    2,882,379    2,882,379          0.0  filter_particles_inside_ball(Particles, Particles, int, int *)                                      
      0.0          587,162         80        7,339.5        7,359.5        6,176        8,224        482.6  void cub::CUB_200700_750_NS::DeviceScanKernel<cub::CUB_200700_750_NS::DeviceScanPolicy<int, cub::CU…
      0.0          174,943         80        2,186.8        2,176.0        1,600        2,592        172.6  void cub::CUB_200700_750_NS::DeviceScanInitKernel<cub::CUB_200700_750_NS::ScanTileState<int, (bool)…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  ---------  --------  --------  ----------  -----------  ------------------
     97.0      207,853,676  2,009  103,461.3   1,344.0       960  34,339,323  1,847,213.5  [CUDA memcpy DtoH]
      3.0        6,344,849  4,003    1,585.0   1,792.0       415      17,760      1,112.0  [CUDA memset]     
      0.0              672      1      672.0     672.0       672         672          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
  1,005.008  4,003     0.251     0.000     0.000     5.000        0.261  [CUDA memset]     
    322.406  2,009     0.160     0.000     0.000    52.900        2.889  [CUDA memcpy DtoH]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]

Generated:
    /content/DSMC/cuda/build/report1.nsys-rep
    /content/DSMC/cuda/build/report1.sqlite

```

#### stats now
```
/content/DSMC/cuda/build
Running case: BALL
INITIALIZATION. Elapsed time: 1313.8282 ms
SIMULATION LOOP. Elapsed time: 78016.3177 ms
step=2000
  NP=13233525
  mean_u=(9.765007e+02, -4.134474e+00, -9.076714e-04)
  T=3.382422e+02
  totalCollisions = 1283579647
ALL PROGRAM FINISHED. Elapsed time: 80506.9921 ms
Generating '/tmp/nsys-report-6eb1.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /content/DSMC/cuda/build/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  --------  -----------  ------------  ----------------------
     99.7   80,349,255,329        810  99,196,611.5  100,139,916.5   302,619  389,458,816  14,972,697.3  poll                  
      0.2      174,820,120        652     268,129.0       13,689.0       332   16,953,493     857,138.2  ioctl                 
      0.0       38,233,689          2  19,116,844.5   19,116,844.5   810,731   37,422,958  25,888,754.0  pthread_rwlock_wrlock 
      0.0        1,893,090         29      65,279.0       11,163.0     7,722    1,216,997     222,893.1  mmap64                
      0.0          963,888         10      96,388.8       49,005.5    29,567      421,640     122,212.2  sem_timedwait         
      0.0          525,429          1     525,429.0      525,429.0   525,429      525,429           0.0  pthread_cond_wait     
      0.0          386,656         47       8,226.7        7,832.0     1,842       16,587       3,016.0  open64                
      0.0          358,343         38       9,430.1        3,523.5     1,788       98,641      17,137.3  fopen                 
      0.0          234,288         31       7,557.7        2,623.0     1,668       64,086      11,715.7  mmap                  
      0.0          189,098         12      15,758.2       17,297.5     2,787       29,893       8,979.6  write                 
      0.0          163,411         39       4,190.0        3,706.0     2,471       11,651       1,605.3  munmap                
      0.0          136,595      2,550          53.6           44.0        35        5,286         107.7  fputc                 
      0.0          131,881          2      65,940.5       65,940.5    60,530       71,351       7,651.6  pthread_create        
      0.0           71,741         31       2,314.2        1,525.0       801       13,570       2,687.5  fclose                
      0.0           45,182         15       3,012.1          217.0        43       34,856       8,873.1  fwrite                
      0.0           38,865         62         626.9          536.0       165        4,699         666.1  fcntl                 
      0.0           36,713         20       1,835.7           46.0        44       35,767       7,986.6  fgets                 
      0.0           27,227          6       4,537.8        4,703.0     1,651        7,317       2,111.0  open                  
      0.0           26,041         15       1,736.1        1,513.0     1,240        3,520         633.4  read                  
      0.0           22,182          2      11,091.0       11,091.0     6,302       15,880       6,772.7  socket                
      0.0           15,519          3       5,173.0        6,258.0     2,660        6,601       2,183.1  pipe2                 
      0.0           14,580          1      14,580.0       14,580.0    14,580       14,580           0.0  connect               
      0.0           14,193          2       7,096.5        7,096.5     2,530       11,663       6,458.0  pthread_cond_broadcast
      0.0            2,871          8         358.9          357.0       258          475          69.7  dup                   
      0.0            1,757          1       1,757.0        1,757.0     1,757        1,757           0.0  bind                  
      0.0              901          1         901.0          901.0       901          901           0.0  listen                
      0.0              714         10          71.4           38.5        38          333          92.2  fflush                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     99.5   78,695,149,930      6,009   13,096,214.0       22,904.0        7,578  740,336,361  20,901,946.3  cudaMemcpy            
      0.2      184,882,828     32,103        5,759.1        4,740.0        3,187      965,389       8,188.8  cudaLaunchKernel      
      0.2      170,485,582          1  170,485,582.0  170,485,582.0  170,485,582  170,485,582           0.0  cudaMemcpyToSymbol    
      0.0       37,898,388      2,003       18,920.8       15,399.0        4,300    2,059,917      57,027.7  cudaMemset            
      0.0       25,255,106         24    1,052,296.1    1,050,691.5        4,287    2,141,566     695,366.7  cudaFree              
      0.0        1,709,330         24       71,222.1       74,666.0        3,410      168,231      40,774.7  cudaMalloc            
      0.0              976          1          976.0          976.0          976          976           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------------------------------------------------------------------------------------
     28.4   22,325,947,517      2,000   11,162,973.8   11,237,492.0    8,762,371   11,477,774    356,228.5  no_time_counter_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)   
     22.8   17,879,886,150      2,000    8,939,943.1    8,958,775.0    8,354,405    9,048,576    102,986.8  move_particles_kernel(Particles, int, curandStateXORWOW *)                                          
     15.6   12,223,833,930      2,000    6,111,917.0    6,041,136.5    5,480,889  108,030,701  2,285,692.6  counting_sort_scatter_kernel(Particles, Particles, int *, int *, int)                               
     12.3    9,670,816,542      2,000    4,835,408.3    4,807,291.0    3,739,653    5,286,539    171,558.2  scatter_and_key_kernel(Particles, Particles, int *, int *, int *, int *, int)                       
     10.9    8,554,934,860     12,001      712,851.8      715,290.0      658,073    6,769,105     57,249.5  generate_particles_in_rect_kernel(Particles, int, int, float, float, float, float, float, float, in…
      3.5    2,756,624,005        100   27,566,240.1   27,563,898.0   26,927,535   28,121,379    240,248.4  accumulate_sampling_kernel(int *, int *, Cell *, Particles)                                         
      3.1    2,458,373,599      2,000    1,229,186.8    1,227,415.0    1,166,360    1,269,812     14,779.7  mark_valid_kernel(Particles, int *, int)                                                            
      1.8    1,384,886,686      4,000      346,221.7      333,421.5        6,305      698,491    339,118.8  void cub::CUB_200700_750_NS::DeviceScanKernel<cub::CUB_200700_750_NS::DeviceScanPolicy<int, cub::CU…
      0.9      730,689,163          1  730,689,163.0  730,689,163.0  730,689,163  730,689,163          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.7      578,138,914      2,000      289,069.5      286,733.5      162,591      359,133     24,600.7  hss_scheme_kernel(unsigned long long *, Particles, int *, int *, curandStateXORWOW *)               
      0.0        7,573,238      4,000        1,893.3        1,888.0        1,600        2,272        133.4  void cub::CUB_200700_750_NS::DeviceScanInitKernel<cub::CUB_200700_750_NS::ScanTileState<int, (bool)…
      0.0        2,895,243          1    2,895,243.0    2,895,243.0    2,895,243    2,895,243          0.0  filter_particles_inside_ball(Particles, Particles, int, int *)                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  --------  --------  --------  ----------  -----------  ------------------
     94.8      215,815,547  4,009  53,832.8   1,312.0       640  37,420,802  1,342,520.1  [CUDA memcpy DtoH]
      3.0        6,811,721  2,000   3,405.9   3,360.0     2,784       4,096        267.3  [CUDA memcpy DtoD]
      2.2        5,106,968  2,003   2,549.7   2,528.0       448      17,184        379.3  [CUDA memset]     
      0.0              672      1     672.0     672.0       672         672          0.0  [CUDA memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
  1,005.000  2,003     0.502     0.500     0.000     5.000        0.102  [CUDA memset]     
  1,000.000  2,000     0.500     0.500     0.500     0.500        0.000  [CUDA memcpy DtoD]
    322.621  4,009     0.080     0.000     0.000    52.934        2.048  [CUDA memcpy DtoH]
      0.000      1     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy HtoD]

Generated:
    /content/DSMC/cuda/build/report1.nsys-rep
    /content/DSMC/cuda/build/report1.sqlite

```
