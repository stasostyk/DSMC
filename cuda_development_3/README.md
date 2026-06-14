# Add slurm deployment for Galileo 100 cluster, Memory layout improvements, and Implement Half-Split-Shuffle Algorithm for parallelize collisions of heavy cells

## Pull Request 14

https://github.com/stasostyk/DSMC/pull/14

After analyzing the kernel for move particles with NSight Compute, I found that there is an extremely high number of uncoalesced calls to memory. This was the case also on most other kernels. 

To increase coalesced accesses, the entire program is refactored to use SoA.  For example:

| Kernel    | AoS | SoA | % speedup |
| -------- | ------- |------- |------- |
| move_particles_kernel | 52,481,083,624    | 44,248,450,579    |18.6%|
| bin_particles_kernel | 47,881,723,230     |39,323,644,699    | 21.8% |

However, the collisions kernel got substantially slower because it randomly collides particle pairs, so for collisions AoS is actually better. **Therefore, the overall execution time remains about the same.** However, the collisions have lots of room for speedup: currently they are only being done as one thread per cell. So, the next PR will reduce collisions, thus actually doing an overall reduction on execution time.

It is quite long, but here is the NSight Compute kernel analysis pre-optimization:

```
Running Ball case.
==PROF== Connected to process 7109 (/content/DSMC/cuda/build/DSMC_ball)
INITIALIZATION. Elapsed time: 814.9884 ms
==PROF== Profiling "move_particles_kernel": 0%....50%....100% - 31 passes
SIMULATION LOOP. Elapsed time: 21774.6405 ms
step=2000
  NP=1700655
  mean_u=(9.718401e+02, -4.069219e+00, -1.970256e-01)
  T=3.428605e+02
  totalCollisions = 162848141
ALL PROGRAM FINISHED. Elapsed time: 22776.7822 ms
==PROF== Disconnected from process 7109
[7109] DSMC_ball@127.0.0.1
  move_particles_kernel(Particle *, int, curandStateXORWOW *) (12322, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         4.98
    SM Frequency                    Mhz       584.99
    Elapsed Cycles                cycle    1,359,259
    Memory Throughput                 %        56.37
    DRAM Throughput                   %        56.37
    Duration                         ms         2.32
    L1/TEX Cache Throughput           %        46.71
    L2 Cache Throughput               %        22.34
    SM Active Cycles              cycle 1,346,062.57
    Compute (SM) Throughput           %        84.53
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

    Section: GPU Speed Of Light Roofline Chart
    OPT   Est. Speedup: 81.99%                                                                                          
          The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The workload achieved     
          close to 0% of this device's fp32 peak performance and 27% of its fp64 peak performance. If Compute Workload  
          Analysis determines that this workload is fp64 bound, consider using 32-bit precision floating point          
          operations to improve its performance. See the Kernel Profiling Guide                                         
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline      
          analysis.                                                                                                     

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte         1.57
    Dropped Samples                sample            0
    Maximum Sampling Interval       cycle       20,000
    # Pass Groups                                    1
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.13
    Executed Ipc Elapsed  inst/cycle         0.13
    Issue Slots Busy               %         3.31
    Issued Ipc Active     inst/cycle         0.13
    SM Busy                        %        85.22
    -------------------- ----------- ------------

    OPT   FP64 is the highest-utilized pipeline (85.2%) based on active cycles, taking into account the rates of its    
          different instructions. It executes 64-bit floating point operations. The pipeline is over-utilized and       
          likely a performance bottleneck. Based on the number of executed instructions, the highest utilized pipeline  
          (85.2%) is FP64. It executes 64-bit floating point operations. Comparing the two, the overall pipeline        
          utilization appears to be caused by frequent, low-latency instructions. See the Kernel Profiling Guide        
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) or hover over the          
          pipeline name to understand the workloads handled by each pipeline. The Instruction Statistics section shows  
          the mix of executed instructions for this workload. Check the Warp State Statistics section for which         
          reasons cause warps to stall.                                                                                 

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s       179.77
    Mem Busy                    %        22.34
    Max Bandwidth               %        56.37
    L1/TEX Hit Rate             %        79.92
    L2 Hit Rate                 %        69.73
    Mem Pipes Busy              %        25.66
    ----------------- ----------- ------------

    Section: Memory Workload Analysis Tables
    OPT   Est. Speedup: 35.52%                                                                                          
          The memory access pattern for global loads from L1TEX might not be optimal. On average, only 7.7 of the 32    
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced global loads.                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 35.68%                                                                                          
          The memory access pattern for global stores to L1TEX might not be optimal. On average, only 7.6 of the 32     
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced global stores.                                     
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 40.96%                                                                                          
          The memory access pattern for local loads from L1TEX might not be optimal. On average, only 3.9 of the 32     
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced local loads.                                       
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 45.16%                                                                                          
          The memory access pattern for local stores to L1TEX might not be optimal. On average, only 1.1 of the 32      
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced local stores.                                      

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         3.42
    Issued Warp Per Scheduler                        0.03
    No Eligible                            %        96.58
    Active Warps Per Scheduler          warp         3.01
    Eligible Warps Per Scheduler        warp         0.04
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 15.47%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 29.3 cycles. This might leave hardware resources underutilized and may lead to    
          less optimal performance. Out of the maximum of 8 warps per scheduler, this workload allocates an average of  
          3.01 active warps per scheduler, but only an average of 0.04 warps were eligible per cycle. Eligible warps    
          are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible   
          warp results in no instruction being issued and the issue slot remains unused. To increase the number of      
          eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp  
          State Statistics and Source Counters sections.                                                                

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        88.07
    Warp Cycles Per Executed Instruction           cycle        88.14
    Avg. Active Threads Per Warp                                16.08
    Avg. Not Predicated Off Threads Per Warp                    15.84
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 15.47%                                                                                          
          On average, each warp of this workload spends 64.1 cycles being stalled waiting for a scoreboard dependency   
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 72.8% of the total average of 88.1 cycles        
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 42.68%                                                                                          
          Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is         
          achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early     
          thread completion, and divergent flow control can significantly lower the number of active threads in a warp  
          per cycle. This workload achieves an average of 16.1 threads being active per cycle. This is further reduced  
          to 15.8 threads per warp due to predication. The compiler may use predication to avoid an actual branch.      
          Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads  
          execute the instructions. Try to avoid different execution paths within a warp when possible.                 

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    44,499.37
    Executed Instructions                           inst    7,119,899
    Avg. Issued Instructions Per Scheduler          inst    44,535.66
    Issued Instructions                             inst    7,125,706
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 16.91%                                                                                          
          This kernel executes 1564468 fused and 1029052 non-fused FP64 instructions. By converting pairs of non-fused  
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP64 performance could be increased by up to 20% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP64 instructions.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 12,322
    Registers Per Thread             register/thread             108
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                 1,024
    Threads                                   thread       1,577,216
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               77.01
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %           50
    Achieved Occupancy                        %        36.63
    Achieved Active Warps Per SM           warp        11.72
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 15.47%                                                                                          
          The difference between calculated theoretical (50.0%) and measured achieved occupancy (36.6%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 15.47%                                                                                          
          The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 8. This kernel's theoretical occupancy (50.0%) is limited by the number of required       
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle 6,526,648.50
    Total DRAM Elapsed Cycles        cycle   92,626,944
    Average L1 Active Cycles         cycle 1,346,062.57
    Total L1 Elapsed Cycles          cycle   54,283,208
    Average L2 Active Cycles         cycle 1,891,431.28
    Total L2 Elapsed Cycles          cycle   63,570,752
    Average SM Active Cycles         cycle 1,346,062.57
    Total SM Elapsed Cycles          cycle   54,283,208
    Average SMSP Active Cycles       cycle 1,303,753.44
    Total SMSP Elapsed Cycles        cycle  217,132,832
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.08
    Branch Instructions              inst      592,084
    Branch Efficiency                   %        82.64
    Avg. Divergent Branches                     272.44
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 72.52%                                                                                          
          This kernel has uncoalesced global accesses resulting in a total of 25235347 excessive sectors (76% of the    
          total 33132282 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source       
          locations. The CUDA Programming Guide                                                                         
          (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional      
          information on reducing uncoalesced device memory accesses.                    
```

And this is post-improvement:

```
Running Ball case.
==PROF== Connected to process 11813 (/content/DSMC/cuda/build/DSMC_ball)
INITIALIZATION. Elapsed time: 866.4490 ms
==PROF== Profiling "move_particles_kernel": 0%....50%....100% - 31 passes
SIMULATION LOOP. Elapsed time: 21519.5034 ms
step=2000
  NP=1699542
  mean_u=(9.717240e+02, -4.172249e+00, -8.871648e-02)
  T=3.430086e+02
  totalCollisions = 162869301
ALL PROGRAM FINISHED. Elapsed time: 22607.5076 ms
==PROF== Disconnected from process 11813
[11813] DSMC_ball@127.0.0.1
  move_particles_kernel(Particles, int, curandStateXORWOW *) (12322, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.00
    SM Frequency                    Mhz       584.98
    Elapsed Cycles                cycle    1,359,449
    Memory Throughput                 %        48.55
    DRAM Throughput                   %        48.55
    Duration                         ms         2.32
    L1/TEX Cache Throughput           %        26.72
    L2 Cache Throughput               %        16.90
    SM Active Cycles              cycle 1,345,741.12
    Compute (SM) Throughput           %        84.36
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing workloads in the Compute Workload Analysis section.                                        

    Section: GPU Speed Of Light Roofline Chart
    OPT   Est. Speedup: 81.93%                                                                                          
          The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The workload achieved     
          close to 0% of this device's fp32 peak performance and 27% of its fp64 peak performance. If Compute Workload  
          Analysis determines that this workload is fp64 bound, consider using 32-bit precision floating point          
          operations to improve its performance. See the Kernel Profiling Guide                                         
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline      
          analysis.                                                                                                     

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte         1.57
    Dropped Samples                sample            0
    Maximum Sampling Interval       cycle       20,000
    # Pass Groups                                    1
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.14
    Executed Ipc Elapsed  inst/cycle         0.14
    Issue Slots Busy               %         3.56
    Issued Ipc Active     inst/cycle         0.14
    SM Busy                        %        85.16
    -------------------- ----------- ------------

    OPT   FP64 is the highest-utilized pipeline (85.2%) based on active cycles, taking into account the rates of its    
          different instructions. It executes 64-bit floating point operations. The pipeline is over-utilized and       
          likely a performance bottleneck. Based on the number of executed instructions, the highest utilized pipeline  
          (85.2%) is FP64. It executes 64-bit floating point operations. Comparing the two, the overall pipeline        
          utilization appears to be caused by frequent, low-latency instructions. See the Kernel Profiling Guide        
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) or hover over the          
          pipeline name to understand the workloads handled by each pipeline. The Instruction Statistics section shows  
          the mix of executed instructions for this workload. Check the Warp State Statistics section for which         
          reasons cause warps to stall.                                                                                 

    Section: Memory Workload Analysis
    ----------------- ----------- ------------
    Metric Name       Metric Unit Metric Value
    ----------------- ----------- ------------
    Memory Throughput     Gbyte/s       155.36
    Mem Busy                    %        16.90
    Max Bandwidth               %        48.55
    L1/TEX Hit Rate             %        73.37
    L2 Hit Rate                 %        55.98
    Mem Pipes Busy              %        25.61
    ----------------- ----------- ------------

    Section: Memory Workload Analysis Tables
    OPT   Est. Speedup: 14.54%                                                                                          
          The memory access pattern for global loads from L1TEX might not be optimal. On average, only 14.6 of the 32   
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced global loads.                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 18.3%                                                                                           
          The memory access pattern for global stores to L1TEX might not be optimal. On average, only 10.1 of the 32    
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced global stores.                                     
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 25.02%                                                                                          
          The memory access pattern for local loads from DRAM might not be optimal. On average, only 3.9 of the 32      
          bytes transmitted per sector are utilized by each thread. This applies to the 95.9% of sectors missed in L2.  
          This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced  
          local loads.                                                                                                  
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 25.83%                                                                                          
          The memory access pattern for local stores to L1TEX might not be optimal. On average, only 1.1 of the 32      
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced local stores.                                      

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         3.67
    Issued Warp Per Scheduler                        0.04
    No Eligible                            %        96.33
    Active Warps Per Scheduler          warp         3.10
    Eligible Warps Per Scheduler        warp         0.04
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 15.64%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 27.3 cycles. This might leave hardware resources underutilized and may lead to    
          less optimal performance. Out of the maximum of 8 warps per scheduler, this workload allocates an average of  
          3.10 active warps per scheduler, but only an average of 0.04 warps were eligible per cycle. Eligible warps    
          are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible   
          warp results in no instruction being issued and the issue slot remains unused. To increase the number of      
          eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp  
          State Statistics and Source Counters sections.                                                                

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        84.49
    Warp Cycles Per Executed Instruction           cycle        84.56
    Avg. Active Threads Per Warp                                17.24
    Avg. Not Predicated Off Threads Per Warp                    16.81
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 15.64%                                                                                          
          On average, each warp of this workload spends 69.5 cycles being stalled waiting for a scoreboard dependency   
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited  
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 82.2% of the total average of 84.5 cycles        
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 40.06%                                                                                          
          Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is         
          achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early     
          thread completion, and divergent flow control can significantly lower the number of active threads in a warp  
          per cycle. This workload achieves an average of 17.2 threads being active per cycle. This is further reduced  
          to 16.8 threads per warp due to predication. The compiler may use predication to avoid an actual branch.      
          Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads  
          execute the instructions. Try to avoid different execution paths within a warp when possible.                 

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst    47,804.61
    Executed Instructions                           inst    7,648,737
    Avg. Issued Instructions Per Scheduler          inst    47,845.11
    Issued Instructions                             inst    7,655,217
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 16.9%                                                                                           
          This kernel executes 1562882 fused and 1028362 non-fused FP64 instructions. By converting pairs of non-fused  
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP64 performance could be increased by up to 20% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP64 instructions.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 12,322
    Registers Per Thread             register/thread             108
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              40
    Stack Size                                                 1,024
    Threads                                   thread       1,577,216
    # TPCs                                                        20
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               77.01
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            8
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %           50
    Achieved Occupancy                        %        37.74
    Achieved Active Warps Per SM           warp        12.08
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 15.64%                                                                                          
          The difference between calculated theoretical (50.0%) and measured achieved occupancy (37.7%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 15.64%                                                                                          
          The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 8. This kernel's theoretical occupancy (50.0%) is limited by the number of required       
          registers.                                                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    5,641,366
    Total DRAM Elapsed Cycles        cycle   92,962,816
    Average L1 Active Cycles         cycle 1,345,741.12
    Total L1 Elapsed Cycles          cycle   54,340,384
    Average L2 Active Cycles         cycle 1,811,778.78
    Total L2 Elapsed Cycles          cycle   63,579,744
    Average SM Active Cycles         cycle 1,345,741.12
    Total SM Elapsed Cycles          cycle   54,340,384
    Average SMSP Active Cycles       cycle 1,304,831.54
    Total SMSP Elapsed Cycles        cycle  217,361,536
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.08
    Branch Instructions              inst      591,533
    Branch Efficiency                   %        82.64
    Avg. Divergent Branches                     272.16
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 55.29%                                                                                          
          This kernel has uncoalesced global accesses resulting in a total of 14589292 excessive sectors (61% of the    
          total 24063391 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source       
          locations. The CUDA Programming Guide                                                                         
          (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional      
          information on reducing uncoalesced device memory accesses.                                                   

## Pull Request 15

https://github.com/stasostyk/DSMC/pull/15

The NTC collision algorithm has a big limitation, which makes it only have one thread per cell. Every random pair can have repeated particles, and they must be collided in series. If we remove possible same-particle collisions or neglect the order of operations, we lose on the stochastic properties that make DSMC work. In terms of theory, these almost neglible events are actually very important. 
1. First, I tried parallelizing while keeping the random generation serial. This was too slow.
2. Then I tried doing dynamic parallelism with child kernels, but could not replicate the stochastic properties of same-particle collisions across threads.
3. Then I found this paper, a PhD thesis from last year which was published in a top tier journal, Springer's Rarefied Gas Dynamics (which is the parent category of DSMC). It is as far as I found the first and only algorithm to parallelize collisions while fully maintaining stochastic properties (so no cheating by omitting same-particle collisions). It does so in an algorithm called "Half Split Shuffle", basically for a few iterations you randomly shuffle the list of particles in a cell, collide the first half with the second half, shuffle and collide again, ... repeat. https://link.springer.com/chapter/10.1007/978-3-032-00094-1_38

The naive version was several times slower than NTC, but then I parallelized it where possible. However, there is still quite some serial work, so often most threads wait for the thread 0 to finish its serial work. Also, it was developed for FPGAs, not for GPUs. I think my implementation is the first ever CUDA Half-Split-Shuffle to exist. It was marginally faster on collision dense problems, and slower on most problems. 

So next optimization: Why not make it hybrid?
4. I made it hybrid, every cell is either handled by NTC or HSS depending on a global threshold. I found the optimal threshold to be 400 particles for 64 thread blocks. Anything below this, NTC handles. Above this, HSS handles. To make the hybrid model easier, instead of only launching each scheme on its respective cells, we launch both schemes on all cells, and return ASAP in the code when we detect the threshold condition. I tested and the overhead of this strategy is super low, so it seems valid.

Now, we have the following results. Basically for problem sizes like 10x10x10 cells, there are many collisions per cell, so we get much better performance, whereas 50x50x50 gets equivalent performance since barely any cells are ran with HSS and the overhead is neglible. So performance gains varies between 1% to 17.5% together with the AoS implementation.

 | Month    | NTC + AoS (ms) | NTC + SoA (ms) | Hybrid NTC/HSS + SoA (ms) | % speedup |
| -------- | ------- |------- |------- | ------- |
| 10x10x10 | 3663.6026    | similar as left   |3022.3062    | 21.2% |
| 20x20x20 | 15271.7155  |similar as left   | 14038.5275 | 8.7% |
| 50x50x50 | 243767.2319 |similar as left   | 243162.5532 | 0% |
```


**Conclusion: Handling collision-dense cells with HSS undoes the SoA penalty at worst, and leads to substantial (17.5%) improvement at best.**

