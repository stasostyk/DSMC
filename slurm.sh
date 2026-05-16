#!/bin/bash
#SBATCH --job-name=dsmc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR/DSMC/cuda"

module purge
module load cuda
module load cmake

nvidia-smi

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . --parallel "$SLURM_CPUS_PER_TASK"

mkdir -p "$SLURM_SUBMIT_DIR/DSMC/cuda/build/profiles"

PROFILE_BASE="${TMPDIR:-/tmp}/dsmc_ball_${SLURM_JOB_ID}"

nsys profile \
  --trace=cuda \
  --stats=true \
  --force-overwrite=true \
  --output="$PROFILE_BASE" \
  ./DSMC_ball

cp "${PROFILE_BASE}"* "$SLURM_SUBMIT_DIR/DSMC/cuda/build/profiles/"