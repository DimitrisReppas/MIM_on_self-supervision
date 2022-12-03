#!/bin/bash
#SBATCH --job-name=k_t007k6l15                         # name of job
#SBATCH -C v100-32g                                        # reserving 32 GB GPUs only
#SBATCH --ntasks=1                                       # total number of GPUs
#SBATCH --ntasks-per-node=1                               # GPUs per node
#SBATCH --nodes=1                                          # reserving n node
#SBATCH --gres=gpu:1                                       # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10                                 # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread                               # hyperthreading is deactivated
#SBATCH --time=20:00:00                                    # maximum execution time requested (HH:MM:SS)
#SBATCH --output=./logfiles/k_t007k6l15log_%j.out                       # name of output file
#SBATCH --error=./logfiles/k_t007k6l15log_%j.error                      # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3


cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.1

srun python ./eval_knn.py --pretrained_weights /gpfsscratch/rech/opx/upl75ty/contrastive/t007k6l15/checkpoint0100.pth --data_path /gpfsdswork/dataset/imagenet/RawImages/
