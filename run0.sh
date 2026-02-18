#!/bin/bash
#SBATCH --job-name=Run0
#SBATCH --output=Paper1_data_experiment/GPU0.log
#SBATCH --error=Paper1_data_experiment/GPU0.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1

echo "loading modules"

. /home/spack/share/spack/setup-env.sh
#spack load py-torch
spack load /j5cepfd
spack load anaconda3

source /usr1/software/miniconda3/etc/profile.d/conda.sh
conda activate /usr1/home/abdulla.fathalla/.aixvipmap/envs/MLEnv

echo "starting script"

python -u Paper1_data_experiment/PointNetMLPJoint/GPU1.py

echo "DONE"
