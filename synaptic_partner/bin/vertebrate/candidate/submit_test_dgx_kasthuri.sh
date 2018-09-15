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

pred_thd_start=-0.5
iteration_start=21500

for ((imultiple=0;imultiple<20;imultiple++));
do

    iteration1=`echo $iteration_start + 500*$imultiple | bc -l`
#     iteration="$(printf "%1d\n" $iteration1)"
#     iteration1= ((iteration_start + 1000*imultiple))
#     iteration="$(printf "%1d\n" $iteration1)"

    model_name=kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased/syn_prune_kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased_${iteration1}.json
    weightname=kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased/sys_prune_kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_unbiased_${iteration1}_weights.h5


    echo model $model_name
    echo weight $weightname
    echo threshold $pred_thd


    THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u test.py  --trial=kasthuri_test_seg_trial_0.3_o100_leaky_f24_160_16_122K --datadir=kasthuri_test_files --imagedir=grayscale_maps2_cropped --predname=ac3_synapse-polarity_full_linear_leaky_f24_316_32_122500.h5 --syn_gtname ac3_syn_groundtruth_cropped.h5  --segname=ac3-seg_m.h5 --seg_gtname ac3_seg_groundtruth_cropped.h5  --inputSize_xy=160 --inputSize_z=16 --modelname $model_name  --weightname $weightname  --cleft_label

done



# end of program
exit 0;
