import time
import glob


import os, sys
os.environ['KERAS_BACKEND']='theano'

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import scipy.misc
import scipy.ndimage
from prepare_data_3d_my_test import *
import pdb
import argparse

from keras.models import Model, Sequential, model_from_json

import keras.activations
import theano

parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--trial', dest='train_trial', action='store', default='trial00', help='trial id')
parser.add_argument('--datadir', dest='datadir', action='store', default='.', help='folder containing date')
parser.add_argument('--imagedir', dest='imagedir', action='store', default='./grayscale_maps', help='image subfolder')
parser.add_argument('--predname', dest='predname', action='store', required=True, help='synaptic polarity prediction')
parser.add_argument('--segname', dest='segname', action='store', required=True, help='segmentation file')
parser.add_argument('--syn_gtname', dest='syn_gtname', action='store', required=True, help='synpatic GT')
parser.add_argument('--seg_gtname', dest='seg_gtname', action='store', default=None, help='segmentation GT')
parser.add_argument('--inputSize_xy', dest='patchSize', action='store', default=None, help='input patch size in xy')
parser.add_argument('--inputSize_z', dest='patchZ', action='store', default=None, help='input patch size in z')
parser.add_argument('--modelname', dest='modelname', action='store', default=None, help='deep net model name')
parser.add_argument('--weightname', dest='weightname', action='store', default=None, help='deep net weight name')
#parser.add_argument('--threshold', dest='threshold', action='store', default=None, help='prediction threshold')
parser.add_argument('--cleft_label', action='store_true', default=False, help='use margin', dest='cleft_only')


if __name__=='__main__':
    
    args = parser.parse_args()
    #pdb.set_trace()
    
    train_trial = args.train_trial
    train_datadir = args.datadir
    train_imagedir = args.imagedir
    train_predname = args.predname
    train_syn_gtname = args.syn_gtname
    train_segname = args.segname
    only_cleft=args.cleft_only	
    #train_seg_gtname = args.seg_gtname

    #train_trial = sys.argv[1]
    #train_datadir = sys.argv[2]
    #train_imagedir = sys.argv[3]
    #train_predname = sys.argv[4]
    #train_gtname = sys.argv[5]
    #train_segname = sys.argv[6]
    
    doFineTune = False

    #patchSize = 192#428# 316# 204
    patchSize = int(args.patchSize)
    patchSize_out = 116 # 340 # 228# 116
    #patchZ = 22
    patchZ = int(args.patchZ)
    patchZ_out = 4


    gen_data = GenerateData(train_trial,train_datadir,train_imagedir,train_predname, train_syn_gtname,train_segname,  patchSize, patchZ)


    #modelname = 'val_seg_trial_0.3_o100/syn_prune_val_seg_trial_0.3_o100_20000.json'
    #weightname = 'val_seg_trial_0.3_o100/sys_prune_val_seg_trial_0.3_o100_20000_weights.h5'

    #modelname = 'val_seg_trial_0.3_o100_10_06_17/syn_prune_val_seg_trial_0.4_o100_21500.json'
    #weightname = 'val_seg_trial_0.3_o100_10_06_17/sys_prune_val_seg_trial_0.4_o100_21500_weights.h5'
    
    #modelname = 'val_seg_trial_0.3_o100_10_06_17/syn_prune_val_seg_trial_0.4_o100_21500.json'
    modelname = args.modelname
    #weightname = 'val_seg_trial_0.3_o100_10_06_17/sys_prune_val_seg_trial_0.4_o100_21500_weights.h5'
    weightname = args.weightname
    
    model = model_from_json(open(modelname).read())
    
    model.load_weights(weightname)
    model.compile(loss='mse', optimizer='Adam')

    #pdb.set_trace()
    ndata = gen_data.get_num_candidates()

    t0 = time.time()    
    
    #results = np.zeros((data_x.shape[0],patchZ_out,patchSize_out,patchSize_out))
    #print "Data_y shape: ", data_y.shape
    #gen_data.create_result_vol()
    
    actual_gt=gen_data.ntrue_positives_inside_margin(only_cleft=only_cleft)
    #pdb.set_trace()
    #pred_thd=-0.05 
    #pred_thd=-0.175
    #pred_thd=-0.175
    #pred_thd=float(args.threshold)
    
    
    #pdb.set_trace()
    res=np.zeros((ndata,5)).astype(np.float32)
    for k in range(ndata):
        
        data = gen_data.get_next_test_sample(k)
        data_x = np.array(data[0])
        
        #data_x = np.reshape(data_x, [1,-1, patchZ, patchSize, patchSize])
        print "{0}th data_x shape: {1}".format(k, data_x.shape)  
        #im_pred = model.predict(x=data_x, batch_size = 1)
        im_pred_array = model.predict(x=data_x, batch_size = data_x.shape[0])
        im_pred = np.mean(im_pred_array)
        gen_data.store_prediction(k, im_pred)
        res[k,:] = [data[1], data[4], im_pred, data[2],data[3]]

    gen_data.save_candidates_with_prediction()

    pred_range=np.array(range(-20,20))/20.
    for pred_thd in pred_range:
        syn_gt_detected=[]
        syn_gt_missed=[]
        syn_false_alarm=[]
        input_positive_detected=[]
        true_positive=0
        false_negative=0
        false_positive=0
        print 'threshold: ',pred_thd

        for k in range(ndata):
            if res[k,0]>0:
                input_positive_detected.append(res[k,1])
	
            if res[k,2]>pred_thd and res[k,0]>0:
                #true_positive            
                gt_id = res[k,1]
                #if gt_id not in syn_gt_detected:
                syn_gt_detected.append(gt_id)
                
            elif res[k,2]<pred_thd and res[k,0]>0:
               #false_negative
                gt_id = res[k,1]
                if gt_id not in syn_gt_missed:
                   syn_gt_missed.append(gt_id)
                #print 'gt_id= {4}, label = {0}, pred = {1} | pre= {2}, post= {3} '.format(data[1], im_pred, data[2], data[3],gt_id)
                
            elif res[k,2]>pred_thd and res[k,0]<0:
               #false_positive
               large_id = res[k,3]*65535 + res[k,4] # change 65535 if this is not large enough
               if large_id not in syn_false_alarm:
                   syn_false_alarm.append(large_id)
                   false_positive=false_positive+1
            #pdb.set_trace()
            #fidw=h5py.File('samples/fn_'+str(k)+'.h5','w')
            #fidw.create_dataset('stack',data=data[0])
            #fidw.close()
                #print 'k= {4}, label = {0}, pred = {1} | pre= {2}, post= {3} '.format(data[1], im_pred, data[2], data[3],k)
            
    
    #print 'false negative: {0}'.format(len(np.setdiff1d(syn_gt_missed,syn_gt_detected)))
    #pdb.set_trace()
        uid_detected = np.unique(syn_gt_detected)
        ntp = len(uid_detected)	
        print 'total gt: {0}'.format(actual_gt)
        print 'input positive : {0}'.format(len(np.unique(input_positive_detected)))
        print 'true positive: {0}'.format(ntp)
        print 'false ngative: {0}  ({1})'.format(actual_gt-ntp, ntp*1.0/actual_gt)
        print 'false positive: {0} ({1})'.format(false_positive, ntp*1.0/(false_positive+ntp))
    

        print '-----------------------------'
    #pdb.set_trace()
        bdt=(res[:,1]> pred_thd).astype(int)
        act=(res[:,0]>0.0).astype(int)
        print 'false negative: {0}'.format(np.sum((bdt==0) * (act==1)))
        print 'false positive: {0}'.format(np.sum((bdt==1) * (act==0)))
        print 'true positive: {0}'.format(np.sum((bdt==1) * (act==1)))
        print 'true negative: {0}'.format(np.sum((bdt==0) * (act==0)))
        #t1=time.time()
    #pdb.set_trace()
    #print "prediction of ({0}, {1}, {2}) took {3} seconds".format(patchZ_out, patchSize_out, patchSize_out, (time.time() - t0))
    #pdb.set_trace()   
    #gen_data.save_to_disk('trn_ecs_synapse_polarity_full_margin_linear_316_32_105000.h5')
    

