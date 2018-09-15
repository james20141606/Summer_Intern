

export LD_LIBRARY_PATH=/n/home05/paragt/cuda/lib64:/usr/local/cuda-8.0/lib64

export LIBRARY_PATH=/n/home05/paragt/cuda/lib64:/usr/local/cuda-8.0/lib64

export PATH=/usr/local/cuda-8.0/bin:$PATH

module load Anaconda

source activate keras_env

THEANO_FLAGS=device=gpu$1,floatX=float32,dnn.enabled=True,dnn.library_path=/n/home05/paragt/cuda/lib64,dnn.include_path=/n/home05/paragt/cuda/include python -u  unet_3d_valid_unnorm_leaky_f24.py --trial=kasthuri_synapse_polarity_full_linear_leaky_f24_316_32 --imagedir=/n/coxfs01/paragt/test_submit/ecs_synapse_polarity_full/grayscale_maps2_ac4/  --gtname=/n/coxfs01/paragt/test_submit/ecs_synapse_polarity_full/ac4_syn_polarity_both_corrected.h5


#THEANO_FLAGS=device=gpu$1,floatX=float32,dnn.enabled=True,dnn.library_path=/n/home05/paragt/cuda/lib64,dnn.include_path=/n/home05/paragt/cuda/include python -u  train_prune_cnn.py --trial train_gt_trial_0.3_o100 --datadir=train_files --imagedir=grayscale_maps2_trn3x6x6 --predname=trn_ecs_synapse_polarity_full_margin_linear_316_32_105000-cropped.h5 --syn_gtname=ecs_syn_gt_trn_db.h5  --seg_gtname=ecs_seg_groundtruth.h5 --segname=ecs_seg_groundtruth.h5

#THEANO_FLAGS=device=gpu$1,floatX=float32,dnn.enabled=True,dnn.library_path=/n/home05/paragt/cuda/lib64,dnn.include_path=/n/home05/paragt/cuda/include python -u  train_prune_cnn.py --trial=val_seg_trial_0.3_o100 --datadir=val_files --imagedir=grayscale_maps2_synapse3x4x4 --predname=val_ecs_synapse_polarity_full_margin_linear_316_32_105000-cropped.h5 --syn_gtname=ecs-syn-val-groundtruth-polarity.h5  --segname=result_ecs-synapse-3x4x4-margin-100K-40000-itr3-thd0.1-xml_m.h5


