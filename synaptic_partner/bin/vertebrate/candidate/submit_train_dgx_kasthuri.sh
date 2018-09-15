#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p seas_dgx1 
#SBATCH --gres=gpu:1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=60000
#SBATCH -t 3-0:00:00
#SBATCH -o ecs_test_316_32_%j.log

module load cuda/9.0-fasrc01
module load cudnn/7.0.3-fasrc02

module load Anaconda

source ~/anaconda2/bin/activate kears_theano

#THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True    python -u  train_prune_cnn_leaky.py --trial kasthuri_train_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased --datadir kasthuri_train_files --imagedir grayscale_maps2_cropped --predname ac4_synapse-polarity_full_linear_leaky_f24_316_32_122500.h5 --syn_gtname ac4_syn_groundtruth_cropped.h5  --seg_gtname ac4_seg_groundtruth_cropped.h5 --segname ac4-seg_m.h5 --inputSize_xy=160 --inputSize_z=16

THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u  train_prune_cnn_leaky.py --trial kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased --datadir kasthuri_val_files --imagedir grayscale_maps2 --predname ac2_synapse-polarity_full_linear_leaky_f24_316_32_122500.h5 --syn_gtname ac2_syn_groundtruth_final.h5  --segname ac2-seg_m.h5 --seg_gtname ac2_seg_groundtruth.h5 --inputSize_xy=160 --inputSize_z=16



# end of program
exit 0;
