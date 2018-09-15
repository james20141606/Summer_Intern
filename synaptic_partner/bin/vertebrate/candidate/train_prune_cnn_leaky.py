import os
os.environ['KERAS_BACKEND']='theano'
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, merge, ZeroPadding2D, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from prepare_data_3d_my import *
import multiprocessing
import sys


import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
import random
import scipy.misc
import h5py
import glob
import argparse


purpose = 'train'
initialization = 'glorot_uniform'

srng = RandomStreams(1234)



def maybe_print(tensor, msg, do_print=False):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor

def proximity_activation(x):
    y = theano.tensor.exp(-1*theano.tensor.square(x))
    z = theano.tensor.switch(y>0.001, y , 0)
    return z

def weighted_mse(y_true, y_pred):
    epsilon=0.000001
    pos_mask = K.cast(y_true >= 0.5, 'float32')
    neg_mask = K.cast(y_true < -0.5, 'float32')
    #y_pred = K.clip(y_pred,epsilon,1-epsilon)
    ## all labels with absolute value less than 0.01 is background
    #pos_mask = K.cast(K.abs(y_true) >= 0.75, 'float32')
    #neg_mask = K.cast(K.abs(y_true) < 0.75, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[:]), 'float32')
    num_pixels = maybe_print(num_pixels, "total ",do_print=True)
    num_pos = maybe_print(K.sum(pos_mask),'npositive ',do_print=True)
    pos_fracs = K.clip((num_pos/num_pixels),0.05, 0.95)
    neg_fracs = K.clip((K.sum(neg_mask) /num_pixels),0.05, 0.95)

    pos_fracs = maybe_print(pos_fracs, "positive fraction",do_print=True)

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight", do_print=True)
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight", do_print=True) #1.25
    
    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error)/1.0

    return K.mean(batch_weighted_mse)

def weighted_crossentropy(y_true, y_pred):
    epsilon = 1.0e-4
    y_pred = K.clip(y_pred,epsilon,1-epsilon)
    # all labels with absolute value less than 0.01 is background
    pos_mask = K.cast(K.abs(y_true) >= 0.75, 'float32')
    neg_mask = K.cast(K.abs(y_true) < 0.75, 'float32')
    
    num_pixels = K.cast(K.prod(K.shape(y_true)[:]), 'float32')
    num_pixels = maybe_print(num_pixels, "total ",do_print=True)
    
    num_pos = maybe_print(K.sum(pos_mask),'npositive ',do_print=True)
    pos_fracs = K.clip((num_pos/num_pixels),0.05, 0.95)
    neg_fracs = K.clip((K.sum(neg_mask) /num_pixels),0.05, 0.95)

    pos_fracs = maybe_print(pos_fracs, "positive fraction",do_print=True)

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight", do_print=True)
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight", do_print=True)
    
    pos_weight = maybe_print(pos_weight / (pos_weight+neg_weight), "positive weight", do_print=True)
    neg_weight = maybe_print(pos_weight / (pos_weight+neg_weight), "negative weight", do_print=True)
    #per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    cost = -K.mean((pos_weight * y_true* K.log(y_pred)) + (neg_weight * (1-y_true) * K.log(1-y_pred)),axis=1)
    mean_loss = cost
    return mean_loss



parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--trial', dest='train_trial', action='store', default='trial00', help='trial id')
parser.add_argument('--datadir', dest='datadir', action='store', default='.', help='folder containing date')
parser.add_argument('--imagedir', dest='imagedir', action='store', default='./grayscale_maps', help='image subfolder')
parser.add_argument('--predname', dest='predname', action='store', required=True, help='synaptic polarity prediction')
parser.add_argument('--segname', dest='segname', action='store', required=True, help='segmentation file')
parser.add_argument('--syn_gtname', dest='syn_gtname', action='store', required=True, help='synpatic GT')
parser.add_argument('--seg_gtname', dest='seg_gtname', action='store', default=None, help='segmentation GT')
parser.add_argument('--inputSize_xy', dest='patchSize', action='store', default=None, help='segmentation GT')
parser.add_argument('--inputSize_z', dest='patchZ', action='store', default=None, help='segmentation GT')
parser.add_argument('--fine_tune', action='store_true', default=False, help='transfer model', dest='doFineTune')


    
if __name__=='__main__':
    
    args = parser.parse_args()
    #pdb.set_trace()
    
    train_trial = args.train_trial
    train_datadir = args.datadir
    train_imagedir = args.imagedir
    train_predname = args.predname
    train_syn_gtname = args.syn_gtname
    train_segname = args.segname
    #train_seg_gtname = args.seg_gtname

    
    doFineTune = args.doFineTune

    #patchSize = 192#428#316#204
    patchSize = int(args.patchSize)#428#316#204
    patchSize_out = 116#340#228#116
    #patchZ = 22
    patchZ = int(args.patchZ)
    patchZ_out = 4
    print 'input patch xy size: ',patchSize
    print 'input patch z size: ',patchZ
    

    gen_data = GenerateData(train_trial,train_datadir,train_imagedir,train_predname, train_syn_gtname,train_segname,  patchSize, patchZ)

    #pdb.set_trace()
    if doFineTune:
        list_of_files = glob.glob(train_trial+'/*.json') 
        latest_json_file = max(list_of_files, key=os.path.getctime)
        print latest_json_file
        jiter=int(os.path.splitext(os.path.basename(latest_json_file))[0].split('_')[-1])
        model = model_from_json(open(latest_json_file).read())

        list_of_files = glob.glob(train_trial+'/*_weights.h5') 
        latest_weight_file = max(list_of_files, key=os.path.getctime)
        print latest_weight_file        
        witer=int(os.path.splitext(os.path.basename(latest_json_file))[0].split('_')[-1])
        if jiter!=witer:
            print 'mismtach json and weights'
            exit(0)
        model.load_weights(latest_weight_file)
        print 'use previous parameters'
        print latest_json_file
        print latest_weight_file
        print
        
        st_epoch=witer+1

    else:

        # input data should be large patches as prediction is also over large patches
        print
        print "==== building network ===="
        print


        print "== BLOCK 1 =="
        input = Input(shape=(4, patchZ, patchSize, patchSize))

        n_initial_kernels = 22
        layer=1
        last_op = Convolution3D(nb_filter=n_initial_kernels, kernel_dim1=3, kernel_dim2=5, kernel_dim3=5, subsample=(1,1,1),init=initialization, border_mode="valid")(input)
        print "layer ", layer, 'shape ', last_op._keras_shape
        #last_op = Convolution3D(nb_filter=n_initial_kernels, kernel_dim1=3, kernel_dim2=5, kernel_dim3=5, subsample=(1,1,1),init=initialization, border_mode="valid", activation='relu')(last_op)
        #print "layer ", layer, 'shape ', last_op._keras_shape
	last_op = PReLU()(last_op)
        last_op = MaxPooling3D(pool_size=(1, 2, 2))(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        layer=2
        last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid")(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape
        #last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=5, kernel_dim3=5, subsample=(1,1,1),init=initialization, border_mode="valid", activation='relu')(last_op)
        #print "layer ", layer, 'shape ', last_op._keras_shape
	last_op = PReLU()(last_op)
        last_op = MaxPooling3D(pool_size=(1, 2, 2))(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape


        layer=3
        last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid")(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape
        #last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid", activation='relu')(last_op)
        #print "layer ", layer, 'shape ', last_op._keras_shape

	last_op = PReLU()(last_op)
        last_op = MaxPooling3D(pool_size=(1, 2, 2))(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape


        layer=4
        last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid")(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape
        #last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid", activation='relu')(last_op)
        #print "layer ", layer, 'shape ', last_op._keras_shape
        

	last_op = PReLU()(last_op)
        last_op = MaxPooling3D(pool_size=(1, 2, 2))(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        layer=5
        last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-2)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid")(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

	last_op = PReLU()(last_op)
        last_op = MaxPooling3D(pool_size=(1, 2, 2))(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        #act1 = ReLU()(act1)
        #layer=6
        #last_op = Convolution3D(nb_filter=n_initial_kernels*(2**(layer-1)), kernel_dim1=3, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),init=initialization, border_mode="valid", activation='relu')(last_op)
        #print "layer ", layer, 'shape ', last_op._keras_shape

        last_op = Flatten()(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        last_op = Dense(1024)(last_op)
        last_op = Dropout(0.5)(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        last_op = Dense(256)(last_op)
        last_op = Dropout(0.5)(last_op)
        print "layer ", layer, 'shape ', last_op._keras_shape

        output__ = Dense(1)(last_op)
        output = Activation('linear')(output__)
            
        model = Model(input=input, output=output)

        st_epoch=0


    learning_rate_init=1e-4
    decay_rate=5e-6
    adm=Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=weighted_mse, optimizer=adm)


    batch_sz = 15


    for epoch in xrange(st_epoch,st_epoch+20100):
        print
        print "iteration: "+str(epoch)
        
        #generate data from different blocks

        data = gen_data.get_3d_sample2(nsamples_batch=batch_sz)

        #pdb.set_trace()
        #fidw=h5py.File('samples/input_sample_'+str(epoch)+'.h5','w')
        #fidw.create_dataset('stack',data=data[0])
        #fidw.create_dataset('label',data=data[1])
        #fidw.close()

        data_x = data[0].astype(np.float32)
        #data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
        data_y = (data[1]).astype(np.float32)
        #data_y = np.reshape(data_y, [-1, 1, patchZ_out, patchSize_out, patchSize_out])

        #pdb.set_trace()
        #### # debug
        ###fidw=h5py.File('sample_'+str(epoch)+'.h5','w')
        ###fidw.create_dataset('image',data=data_x[0,0,...])
        ###fidw.create_dataset('label',data=data_y[0,...])
        ###fidw.close()
        
        print "Data_x shape: ", data_x.shape
        print "Data_y shape: ", data_y.shape
        current_lr = (learning_rate_init /(1. + epoch*decay_rate)) 
        K.set_value(model.optimizer.lr, current_lr)

        print "current learning rate", model.optimizer.lr.get_value()


        model.fit(data_x, data_y, batch_size=batch_sz, nb_epoch=1)
        
        
        


        if (epoch%500)==0:
            if not os.path.exists(train_trial):
                os.makedirs(train_trial)
            json_string = model.to_json()
            open(os.path.join(train_trial,'syn_prune_'+train_trial+'_'+str(epoch)+'.json'), 'w').write(json_string)
            model.save_weights(os.path.join(train_trial,'sys_prune_'+train_trial+'_'+str(epoch)+'_weights.h5'))





