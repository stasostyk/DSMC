#!/bin/bash
#SBATCH --job-name=gtest-astralog
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

set -e

cd "$SLURM_SUBMIT_DIR/DSMC/cuda"

module purge
module load cuda
module load cmake
module load openmpi

nvidia-smi

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build . --parallel "$SLURM_CPUS_PER_TASK"

PROFILE_BASE="${TMPDIR:-/tmp}/dsmc_sphere_${SLURM_JOB_ID}"

nsys profile \
  --stats=true \
  --force-overwrite=true \
  --output="$PROFILE_BASE" \
  ./DSMC_single_gpu SPHERE

cp ./*.dat "$SLURM_SUBMIT_DIR"/

cp "$PROFILE_BASE".* "$SLURM_SUBMIT_DIR"/