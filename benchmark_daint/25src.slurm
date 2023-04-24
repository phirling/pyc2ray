#!/bin/bash -l
#SBATCH --job-name=pyc2ray
#SBATCH --account=sk015
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=03:00:00

#SBATCH --output=logs/c2ray.%j.out
#SBATCH --error=logs/c2ray.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=patrick.hirling@epfl.ch

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CRAY_CUDA_MPS=1

srun python c2ray_128_25src.py
