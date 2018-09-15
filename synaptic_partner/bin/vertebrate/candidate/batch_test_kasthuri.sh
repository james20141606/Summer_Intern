
datasz=$1
startsz=$2
query_sz=$3
max_iter=$4
nclass=$5
belief=1.0
MAX_NBR_DIST=2.5

TR_DTST_ID=2
TR_DTST=250-${TR_DTST_ID}

CLFR_DIR=/groups/flyem/proj/cluster/toufiq/goo/output/${TR_DTST}
# CLFR_DIR=/groups/scheffer/home/paragt/pixel_classify/data/${TR_DTST}

max_edges=10000

TST_DTST=$6
TST_NAME=$7

mito_thd=0.35

pred_thd_start=-1.0
iteration_start=10000

for ((imultiple=0;imultiple<21;imultiple++)); 
do
    
    iteration1=`echo $iteration_start + 1000*$imultiple | bc -l`
#     iteration="$(printf "%1d\n" $iteration1)"
#     iteration1= ((iteration_start + 1000*imultiple))
#     iteration="$(printf "%1d\n" $iteration1)"
    
    for ((multiple=0;multiple<41;multiple++)); 
    do 
        pred_thd1=`echo $pred_thd_start + 0.05*$multiple | bc -l`
        pred_thd="$(printf "%.2f\n" $pred_thd1)"
        
        model_name=kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K/syn_prune_kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_${iteration1}.json
        weightname=kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K/sys_prune_kasthuri_val_seg_trial_0.3_o100_leaky_f24_160_16_122K_${iteration1}_weights.h5
        
        echo model $model_name
        echo weight $weightname
        echo threshold $pred_thd
        
        THEANO_FLAGS=device=gpu2,floatX=float32,dnn.enabled=True,dnn.library_path=/n/home05/paragt/cuda/lib64,dnn.include_path=/n/home05/paragt/cuda/include python -u test.py  --trial=kasthuri_test_seg_trial_0.3_o100_leaky_f24_160_16_122K --datadir=kasthuri_test_files --imagedir=grayscale_maps2_cropped --predname=ac3_synapse-polarity_full_linear_leaky_f24_316_32_122500.h5 --syn_gtname ac3_syn_groundtruth_cropped.h5  --segname=ac3-seg_m.h5 --seg_gtname ac3_seg_groundtruth_cropped.h5  --inputSize_xy=160 --inputSize_z=16 --threshold=$pred_thd --modelname $model_name  --weightname $weightname  --cleft_label 
        
 
    done
done




