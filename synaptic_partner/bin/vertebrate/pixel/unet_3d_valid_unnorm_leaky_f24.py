import os
os.environ['KERAS_BACKEND']='theano'
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, concatenate,merge, ZeroPadding2D, Dropout, Lambda
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
import argparse





rng = np.random.RandomState(7)

learning_rate = 0.0001
momentum = 0.99

patchSize = 316#428#316#204
patchSize_out = 228#340#228#116
patchZ = 32
patchZ_out = 4
cropSize = (patchSize-patchSize_out)/2
csZ = (patchZ-patchZ_out)/2

weight_decay = 0.
weight_class_1 = 1.

patience = 100
patience_reset = 100 
doBatchNormAll = False

purpose = 'train'
initialization = 'glorot_uniform'
fileprefix = '3D_unet'
numKernel = 22

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
    #y_pred = K.clip(y_pred,epsilon,1-epsilon)
    # all labels with absolute value less than 0.01 is background
    pos_mask = K.cast(K.abs(y_true) >= 0.01, 'float32')
    neg_mask = K.cast(K.abs(y_true) < 0.01, 'float32')
    # emphasize on the number of labels > delta, here, delta =0.99 
    pos_mask2 = K.cast(K.abs(y_true) >= 0.99, 'float32')
    neg_mask2 = K.cast(K.abs(y_true) < 0.99, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask2)/num_pixels),0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask2) /num_pixels),0.01, 0.99)

    pos_fracs = maybe_print(pos_fracs, "positive fraction",do_print=True)

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight", do_print=True)
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight", do_print=True)
    
    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error)/2.0

    return K.mean(batch_weighted_mse)


def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False, downsampleZ=False, thickness1=1, thickness2=1):
    # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
    # All are valid area, not same
    act1 = Convolution3D(filters=nb_filter, kernel_dim1=thickness1, kernel_dim2=3, kernel_dim3=3, subsample=(1, 1, 1),
                         init=initialization, padding="valid")(input)
    act1 = PReLU()(act1)

    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    act2 = Convolution3D(filters=nb_filter, kernel_dim1=thickness2, kernel_dim2=3, kernel_dim3=3, subsample=(1, 1, 1),
                         init=initialization, padding="valid")(act1)
    act2 = PReLU()(act2)

    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)

    if doDropout:
        act2 = Dropout(0.5)(act2)

    if doPooling:
        # now downsamplig with maxpool
        if downsampleZ:
            pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act2)

        else:
            pool1 = MaxPooling3D(pool_size=(1, 2, 2))(act2)

    else:
        pool1 = act2

    return (act2, pool1)


# need to define lambda layer to implement cropping
def crop_layer(x, cs, csZ):
    cropSize = cs
    if csZ == 0:
        return x[:,:,:,cropSize:-cropSize, cropSize:-cropSize]
    else:	
        return x[:,:,csZ:-csZ,cropSize:-cropSize, cropSize:-cropSize]


def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False, upsampleZ=False, thickness1=1, thickness2=1):
    print "This is unet_block_up"
    print "input ", input._keras_shape
    # upsampling
    if upsampleZ:
        up_sampled = UpSampling3D(size=(2,2,2))(input)
    else:
        up_sampled = UpSampling3D(size=(1,2,2))(input)
    print "upsampled ", up_sampled._keras_shape
    # up-convolution
    conv_up = Convolution3D(filters=nb_filter, kernel_dim1=1, kernel_dim2=2, kernel_dim3=2, subsample=(1, 1, 1),
                            init=initialization, padding="same")(up_sampled)
    conv_up = PReLU()(conv_up)

    print "up-convolution ", conv_up._keras_shape
    # concatenation with cropped high res output
    # this is too large and needs to be cropped
    print "to be merged with ", down_block_out._keras_shape

    cropSize = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
    csZ      = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
    if cropSize>0:
    # input is a tensor of size (batchsize, channels, thickness, width, height)
        down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize,"csZ":csZ})(down_block_out)
    else: 
        down_block_out_cropped = down_block_out
        
    print "cropped layer size: ", down_block_out_cropped._keras_shape
    #merged = merge([conv_up, down_block_out_cropped], mode='concat', concat_axis=1)
    merged = concatenate([conv_up, down_block_out_cropped], axis=1)
    print "merged ", merged._keras_shape
    act1 = Convolution3D(filters=nb_filter, kernel_dim1=thickness1, kernel_dim2=3, kernel_dim3=3, subsample=(1, 1, 1),
                         init=initialization, padding="valid")(merged)
    act1 = PReLU()(act1)

    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    print "conv1 ", act1._keras_shape
    act2 = Convolution3D(filters=nb_filter, kernel_dim1=thickness2, kernel_dim2=3, kernel_dim3=3, subsample=(1, 1, 1),
                         init=initialization, padding="valid")(act1)
    act2 = PReLU()(act2)

    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)

    print "conv2 ", act2._keras_shape

    return act2

parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--trial', dest='train_trial', action='store', default='trial00', help='trial id')
parser.add_argument('--imagedir', dest='train_imagedir', action='store', help='image subfolder')
parser.add_argument('--gtname', dest='train_gtname', action='store', help='GT')
parser.add_argument('--ft', dest='fine_tune', help='retrain on previous model')
parser.add_argument('--epoch', dest='epochs', help='epoch')


    
if __name__=='__main__':
    
    args = parser.parse_args()
    
    
    train_trial = args.train_trial#.argv[1]
    train_imagedir = args.train_imagedir #sys.argv[2]
    train_gtname = args.train_gtname #sys.argv[3]
    loadPrevModel = int(args.fine_tune)
    print loadPrevModel
    # input data should be large patches as prediction is also over large patches
    if loadPrevModel:
        print ('fine tune on previous model')
        list_of_files = glob.glob(train_trial+'/*.json') 
        max_iter=0
        for filepath in list_of_files:
            jiter=int(os.path.splitext(os.path.basename(filepath))[0].split('_')[-1])
            if max_iter<jiter: max_iter=jiter

        print 'max iter: ',max_iter
        
        latest_json_file = os.path.join(train_trial,fileprefix+'_'+train_trial+'_'+str(max_iter)+'.json')
        print latest_json_file
        model = model_from_json(open(latest_json_file).read())

        latest_weight_file = os.path.join(train_trial,fileprefix+'_'+train_trial+'_'+str(max_iter)+'_weights.h5')
        print latest_weight_file        
        model.load_weights(latest_weight_file)
        
        print 'use previous parameters'
        print latest_json_file
        print latest_weight_file
        print
        
        st_epoch=max_iter+1

    else:
        print
        print "==== building network ===="
        print

        print "== BLOCK 1 =="
        input = Input(shape=(1, patchZ, patchSize, patchSize))
        print "input  ", input._keras_shape
        block1_act, block1_pool = unet_block_down(input=input, nb_filter=numKernel, doBatchNorm=doBatchNormAll, thickness1=3, thickness2=3, downsampleZ=False)
        print "block1 ", block1_pool._keras_shape

        print "== BLOCK 2 =="
        block2_act, block2_pool = unet_block_down(input=block1_pool, nb_filter=numKernel*2, doBatchNorm=doBatchNormAll, thickness1=3, thickness2=3, downsampleZ=False)
        print "block2 ", block2_pool._keras_shape

        print "== BLOCK 3 =="
        block3_act, block3_pool = unet_block_down(input=block2_pool, nb_filter=numKernel*4, doBatchNorm=doBatchNormAll, thickness1=3, thickness2=3, downsampleZ=False)
        print "block3 ", block3_pool._keras_shape

        print "== BLOCK 4 =="
        block4_act, block4_pool = unet_block_down(input=block3_pool, nb_filter=numKernel*8, doBatchNorm=doBatchNormAll, thickness1=3, thickness2=3, doPooling=False, downsampleZ=False)
        print "block4 ", block4_pool._keras_shape

        ##print "== BLOCK 5 =="
        ##print "#no pooling for the bottom layer"
        ##block5_act, block5_pool = unet_block_down(input=block4_pool, nb_filter=numKernel*16, doPooling=False, doBatchNorm=doBatchNormAll, thickness1=3, thickness2=3)
        ##print "block5 ", block5_pool._keras_shape

        ##print
        ##print "=============="
        ##print

        ##print "== BLOCK 4 UP =="
        ##block4_up = unet_block_up(input=block5_act, nb_filter=numKernel*8, down_block_out=block4_act, doBatchNorm=doBatchNormAll, upsampleZ=False, thickness1=3, thickness2=3)
        ##print
        print "== BLOCK 3 UP =="
        block3_up = unet_block_up(input=block4_act,  nb_filter=numKernel*4, down_block_out=block3_act, doBatchNorm=doBatchNormAll, upsampleZ=False, thickness1=3, thickness2=3)
        print
        print "== BLOCK 2 UP =="
        block2_up = unet_block_up(input=block3_up,  nb_filter=numKernel*2, down_block_out=block2_act, doBatchNorm=doBatchNormAll, upsampleZ=False,thickness1=3, thickness2=3)
        print
        print "== BLOCK 1 UP =="
        block1_up = unet_block_up(input=block2_up,  nb_filter=numKernel*1, down_block_out=block1_act, doBatchNorm=doBatchNormAll, upsampleZ=False,thickness1=3, thickness2=3)
        print
        #print "== 1x1 convolution =="
        #output = Convolution3D(filters=1, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=(1,1,1),
                                #init=initialization, activation='sigmoid',padding="valid")(block1_up)

        print "== 1x1 convolution =="
        output = Convolution3D(filters=1, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=(
            1, 1, 1), init=initialization, activation='linear', padding="valid")(block1_up)

        print "output ", output._keras_shape
        #output_flat = Flatten()(output)
        #print "output flat ", output_flat._keras_shape
        print

            
        model = Model(inputs=input, outputs=output)
        st_epoch = 0

    learning_rate_init=1e-4
    decay_rate=5e-6
    adm=Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=weighted_mse, optimizer=adm)

    gen_data = GenerateData(train_imagedir,train_gtname)

    best_val_loss_so_far = 0

    patience_counter = 0
    for epoch in xrange(st_epoch,st_epoch+300000):
        print
        print "iteration: "+str(epoch)
        
        #generate data from different blocks

        data = gen_data.get_3d_sample(nsamples_patch=1, nsamples_block=1, doAugmentation=True, patchSize=patchSize, patchSize_out=patchSize_out, patchZ=patchZ, patchZ_out=patchZ_out)

        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
        data_y = data[1].astype(np.float32)
        data_y = np.reshape(data_y, [-1, 1, patchZ_out, patchSize_out, patchSize_out])

        ###pdb.set_trace()
        #### # debug
        ###fidw=h5py.File('sample_'+str(epoch)+'.h5','w')
        ###fidw.create_dataset('image',data=data_x[0,0,...])
        ###fidw.create_dataset('label',data=data_y[0,...])
        ###fidw.close()
        
        print "Data_x shape: ", data_x.shape
        print "Data_y shape: ", data_y.shape
    #   print "current learning rate: ", model.optimizer.lr.get_value()
        for k in range(data_x.shape[0]):
            print "The", k, "th input for this round."
            X = data_x[k:k+1]
            Y = data_y[k:k+1]
            
        
            #iteration_number = model.optimizer.iterations.get_value()
            current_lr = (learning_rate_init /(1. + epoch*decay_rate)) 
            
            K.set_value(model.optimizer.lr, current_lr)

            print "current learning rate", model.optimizer.lr.get_value()

            #pdb.set_trace()

            model.fit(X, Y, batch_size=1, epochs=int(args.epochs))



        if (epoch%100)==0:
            if not os.path.exists(train_trial):
                os.makedirs(train_trial)
                
            json_string = model.to_json()
            open(os.path.join(train_trial,fileprefix+'_'+train_trial+'_'+str(epoch)+'.json'), 'w').write(json_string)
            model.save_weights(os.path.join(train_trial,fileprefix+'_'+train_trial+'_'+str(epoch)+'_weights.h5'))

