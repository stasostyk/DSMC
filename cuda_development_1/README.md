# Naive CUDA implementation

## Pull Request 8

https://github.com/stasostyk/DSMC/pull/8

Created a naive CUDA implementation, basically all parts in simulation are calling kernels. Right now everything is done in the most simple way (I just in a straightforward manner took the serial code and moved it to GPU kernels). Lots of optimizations are possible, starting from fixing memory management (in my opinion now there are too many `cudaMemcpy` which most likely are not neccessary).

The statistics (time and profiling) can be found in this branche's notebook: https://github.com/stasostyk/DSMC/blob/cuda_naive/cuda/colab_run.ipynb   but I will paste it here as well:
- params: 
  - NX=NY=NZ=10
  - particlesPerCellTarget = 200
  - nSteps = 2000
- SERIAL BALL case
  - INITIALIZATION. Elapsed time: 55.2708 ms
  - SIMULATION LOOP. Elapsed time: 117706.2772 ms
  - ALL PROGRAM FINISHED. Elapsed time: 117772.6901 ms
- SERIAL WING case
  - INITIALIZATION. Elapsed time: 57.6466 ms
  - SIMULATION LOOP. Elapsed time: 107753.6843 ms
  - ALL PROGRAM FINISHED. Elapsed time: 107821.6167 ms
- CUDA BALL case
  - INITIALIZATION. Elapsed time: 342.0782 ms
  - SIMULATION LOOP. Elapsed time: 219792.0004 ms
  - ALL PROGRAM FINISHED. Elapsed time: 220147.3259 ms
- CUDA WING case
  - INITIALIZATION. Elapsed time: 205.3593 ms
  - SIMULATION LOOP. Elapsed time: 218552.1742 ms
  - ALL PROGRAM FINISHED. Elapsed time: 218777.8221 ms

### NSYS profiling info for CUDA BALL case
```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -------------  ---------  -----------  ------------  ----------------------
     99.7  224,325,161,828      2,244  99,966,649.7  100,165,243.0    331,628  595,128,867  12,378,101.3  poll                  
      0.3      729,371,857     14,547      50,139.0       33,061.0        508   20,952,935     221,269.8  ioctl                 
      0.0        5,605,403          2   2,802,701.5    2,802,701.5    421,524    5,183,879   3,367,493.5  pthread_rwlock_wrlock 
      0.0        4,551,077          1   4,551,077.0    4,551,077.0  4,551,077    4,551,077           0.0  pthread_cond_wait     
      0.0        2,748,732         29      94,783.9       12,518.0     10,690    2,020,510     371,405.0  mmap64                
      0.0          470,950         39      12,075.6        4,537.0      1,437       86,543      17,906.8  fopen                 
      0.0          450,020         12      37,501.7       27,811.0      2,269      144,595      44,147.3  write                 
      0.0          418,355         47       8,901.2        8,801.0      3,365       15,395       2,591.7  open64                
      0.0          239,930         16      14,995.6        7,869.5      3,037       99,584      23,276.5  mmap                  
      0.0          223,777         32       6,993.0        1,575.0      1,161       73,142      16,410.6  fclose                
      0.0          107,565          2      53,782.5       53,782.5     46,580       60,985      10,185.9  pthread_create        
      0.0           68,413         31       2,206.9           66.0         43       54,147       9,693.4  fwrite                
      0.0           57,798         20       2,889.9           72.0         60       56,311      12,574.0  fgets                 
      0.0           56,271          2      28,135.5       28,135.5     25,967       30,304       3,066.7  sem_timedwait         
      0.0           38,945         62         628.1          677.0        280        1,136         186.8  fcntl                 
      0.0           38,016          6       6,336.0        6,273.0      3,010        8,944       2,347.8  open                  
      0.0           37,574          6       6,262.3        6,116.0      5,015        7,990       1,068.0  munmap                
      0.0           21,950          2      10,975.0       10,975.0      7,257       14,693       5,258.0  socket                
      0.0           21,216         15       1,414.4        1,102.0        668        3,864         896.5  read                  
      0.0           16,929          3       5,643.0        5,838.0      2,761        8,330       2,789.6  pipe2                 
      0.0           13,882          1      13,882.0       13,882.0     13,882       13,882           0.0  connect               
      0.0            6,338          2       3,169.0        3,169.0      2,074        4,264       1,548.6  pthread_cond_broadcast
      0.0            4,657        110          42.3           36.0         35          182          16.3  fputc                 
      0.0            3,444          8         430.5          434.0        316          581          88.5  dup                   
      0.0            2,399          1       2,399.0        2,399.0      2,399        2,399           0.0  bind                  
      0.0            1,186          1       1,186.0        1,186.0      1,186        1,186           0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -----------  -----------  --------  -----------  -----------  ----------------------
     99.2  221,841,109,145     78,803  2,815,135.3  3,544,432.0     8,809   25,807,486  2,345,882.3  cudaMemcpy            
      0.4      847,014,592     24,006     35,283.5      6,820.0     3,947  198,652,425  1,283,900.4  cudaMalloc            
      0.2      517,536,041     24,006     21,558.6     10,156.0     4,257    4,283,255     52,506.5  cudaFree              
      0.2      339,567,964     22,102     15,363.7     13,844.0     5,839    3,473,504     31,203.7  cudaLaunchKernel      
      0.1      125,142,638      2,000     62,571.3     61,261.5     9,080      164,590      8,218.8  cudaDeviceSynchronize 
      0.0       41,823,240      4,000     10,455.8      9,924.5     6,463      182,091      5,356.4  cudaMemset            
      0.0            1,745          1      1,745.0      1,745.0     1,745        1,745          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     86.5    7,139,330,207      2,000   3,569,665.1   3,555,252.5   2,966,839   4,308,053    150,210.6  no_time_counter_scheme_kernel(int *, Particle *, int *, int *, double, double, Config *, curandStat…
      3.6      299,550,224      2,000     149,775.1     149,884.0     143,069     154,588      1,348.3  filter_particles_out_of_bounds(Config *, Particle *, Particle *, int, int *)                        
      3.5      292,500,675     12,001      24,373.0      23,904.0      21,567     352,984      4,115.1  generate_particles_in_rect_kernel(Particle *, Config *, int, int, double, double, double, double, d…
      3.1      256,289,782      2,000     128,144.9     125,085.0     118,397     290,137     16,459.0  move_particles_kernel(Particle *, Config *, int, curandStateXORWOW *)                               
      1.8      145,353,325      2,000      72,676.7      72,030.0      68,158     113,565      4,161.9  bin_particles_kernel(Particle *, int *, int *, int, double, double, double)                         
      1.2       97,918,271        100     979,182.7     972,326.5     947,720   1,066,340     24,127.3  accumulate_sampling_kernel(int *, int *, Cell *, Particle *)                                        
      0.2       15,633,115          1  15,633,115.0  15,633,115.0  15,633,115  15,633,115          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)                                                 
      0.0        3,156,412      2,000       1,578.2       1,567.0       1,503       2,880        144.9  reset_cell_count_kernel(int *)                                                                      

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  ------  -----------  -----------  --------  ----------  -----------  ------------------
     55.1  109,813,600,788  46,402  2,366,570.4  1,443,419.0       351  15,770,480  2,481,333.2  [CUDA memcpy HtoD]
     44.9   89,497,574,390  32,401  2,762,185.6  3,627,616.0       800  23,491,478  1,998,377.8  [CUDA memcpy DtoH]
      0.0        1,504,474   4,000        376.1        384.0       351       1,760         46.0  [CUDA memset]     

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 -----------  ------  --------  --------  --------  --------  -----------  ------------------
 515,247.856  46,402    11.104     8.000     0.000    24.000       11.484  [CUDA memcpy HtoD]
 487,662.778  32,401    15.051    24.000     0.000    24.000       10.478  [CUDA memcpy DtoH]
       0.016   4,000     0.000     0.000     0.000     0.000        0.000  [CUDA memset]     
``` 