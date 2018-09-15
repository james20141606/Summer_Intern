#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=60000
#SBATCH -t 3-00:00:00
#SBATCH -o ecs_synapse_multiclass_f24_316_32_%j.log


module load Anaconda

source ~/anaconda2/bin/activate kears_theano


iter=150000



python generate_proposals.py  --trial test_seg_trial_0.3_o100_leaky_f24_160_16 --datadir test_files --imagedir grayscale_maps2_tst4x6x6 --predname test_ecs_synapse_polarity_full_margin_linear_leaky_f24_316_32_196000-cropped.h5  --syn_gtname ecs-syn-tst-groundtruth-polarity.h5  --segname result_ecs-4x6x6-100K-40000-itr3-thd0.1_xml_m.h5  --seg_gtname seg_groundtruth0.h5  --inputSize_xy=160 --inputSize_z=16


# end of program
exit 0;
