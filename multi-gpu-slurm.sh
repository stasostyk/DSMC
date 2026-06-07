#!/bin/bash
#SBATCH --job-name=dsmc-multi-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j-task_%t.out
#SBATCH --error=logs/%x-%j-task_%t.err
#SBATCH --gpus-per-task=1

cd "$SLURM_SUBMIT_DIR/cuda"

module purge
module load cuda/11.5.0
module load openmpi
module load cmake

export NCCL_DEBUG=INFO


cd "$SLURM_SUBMIT_DIR/cuda"
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_CUDA_ARCHITECTURES=70

make -j$(nproc)

nvidia-smi --query-gpu=index,memory.used,memory.total \
           --format=csv -l 1 > "$SLURM_SUBMIT_DIR/logs/gpu_mem.log" &
MONITOR_PID=$!

#srun --ntasks=1 nvidia-smi
#  srun nsys profile --trace=cuda --stats=true --output="$HOME/nsys-out" ./DSMC BALL
srun ./DSMC_mpi BALL

#srun --ntasks=2 --gpus-per-task=1 \
    #bash -c 'if [ $SLURM_PROCID -eq 0 ]; then
    #    nsys profile \
            #--output=$SLURM_SUBMIT_DIR/logs/profile_rank0 \
            #--trace=cuda,mpi \
            #--force-overwrite=true \
            #./DSMC BALL
    #else
        #./DSMC BALL
    #fi'

# After srun, copy the scratch files if nsys still fails to write directly
#mkdir -p cuda/logs
#cp /scratch_local/slurm_job.${SLURM_JOB_ID}/nsys-report-*.qdstrm $SLURM_SUBMIT_DIR/logs/ 2>/dev/null || true

kill $MONITOR_PID

