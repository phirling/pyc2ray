#!/bin/bash -l
#SBATCH --job-name=244Mpc
#SBATCH --account=sk015
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=patrick.hirling@epfl.ch

source ~/astro_env/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'

srun python run_test.py parameters.yml
