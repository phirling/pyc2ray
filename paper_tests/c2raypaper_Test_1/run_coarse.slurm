#!/bin/bash -l
#SBATCH --job-name=pyc2ray
#SBATCH --account=sk015
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=02:00:00

#SBATCH --output=logs/c2ray.%j.out
#SBATCH --error=logs/c2ray.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=patrick.hirling@epfl.ch

module purge
module load gcc/9.3.0
module load nvhpc/21.5

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CRAY_CUDA_MPS=1
export LD_LIBRARY_PATH=/users/phirling/miniconda3/envs/ion/lib:$LD_LIBRARY_PATH

srun python coarse_time.py
