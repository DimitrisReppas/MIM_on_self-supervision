#!/bin/bash
#SBATCH --job-name=train_test_exp                           # name of job
#SBATCH -C v100-32g                                        # reserving 32 GB GPUs only
#SBATCH --ntasks=4                                       # total number of GPUs
#SBATCH --ntasks-per-node=2                               # GPUs per node
#SBATCH --nodes=2                                          # reserving n node
#SBATCH --gres=gpu:2                                       # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10                                 # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread                               # hyperthreading is deactivated
#SBATCH --time=20:00:00                                    # maximum execution time requested (HH:MM:SS)
#SBATCH --output=./logfiles/log_%j.out                       # name of output file
#SBATCH --error=./logfiles/log_%j.error                      # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t3


cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.1

srun python ./multimask_main_ibot.py --act_in_head gelu --arch vit_small --clip_grad 3.0 --data_path /gpfsdswork/dataset/imagenet/RawImages/train/ --epochs 100 --freeze_last_layer 1 --global_crops_number 2 --global_crops_scale 0.25 1.0 --local_crops_number 6 --local_crops_scale 0.05 0.25  --lr 0.0005 --momentum_teacher 0.99 --norm_last_layer False --num_workers 10 --optimizer adamw  --out_dim 8192 --output_dir /gpfsstore/rech/opx/upl75ty/ --patch_out_dim 8192 --patch_size 16 --pred_ratio 0 0.3 --pred_ratio_var 0 0.2 --pred_shape block --pred_start_epoch=0 --saveckp_freq 5 --seed 0 --shared_head True --teacher_patch_temp 0.07 --teacher_temp 0.07  --use_fp16 True --use_masked_im_modeling True --warmup_epochs 10 --warmup_teacher_patch_temp 0.04 --warmup_teacher_temp 0.04 --warmup_teacher_temp_epochs 1 --weight_decay 0.04 --weight_decay_end 0.4 --batch_size_per_gpu 16
