import time
import glob


import os
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

patchSize = 316
patchSize_out = 228
patchZ = 32
patchZ_out = 4

def proximity_activation(x):
    y = theano.tensor.exp(-1*theano.tensor.square(x))
    z = theano.tensor.switch(y>0.001, y , 0)
    return z


parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--imagedir', dest='imagedir', action='store', default='./grayscale_maps', help='image subfolder')
parser.add_argument('--savename', dest='savename', action='store', required=True, help='synaptic polarity prediction')
parser.add_argument('--modelname', dest='modelname', action='store', required=True, help='segmentation file')
parser.add_argument('--weightname', dest='weightname', action='store', required=True, help='synpatic GT')


if __name__=="__main__":
    
    
    args = parser.parse_args()
    #pdb.set_trace()
    
    imagedir = args.imagedir
    savename = args.savename
    modelname = args.modelname
    weightname = args.weightname
    
    gen_data=GenerateData(imagedir, patchSize, patchSize_out, patchZ, patchZ_out)


    #model = model_from_json(f.read(), custom_objects={'my_custom_activation': my_custom_activation})

    model = model_from_json(open(modelname).read())
    
    model.load_weights(weightname)
    model.compile(loss='mse', optimizer='Adam')

    #pdb.set_trace()
    ndata = gen_data.compute_test_sample_indices()

    t0 = time.time()    
    
    #results = np.zeros((data_x.shape[0],patchZ_out,patchSize_out,patchSize_out))
    #print "Data_y shape: ", data_y.shape
    gen_data.create_result_vol()
    
    #pdb.set_trace()
    for k in range(ndata):
        data = gen_data.get_next_test_sample(k)
        data_x = data.astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
        print "{0}th data_x shape: {1}".format(k, data_x.shape)
	    
        im_pred = model.predict(x=data_x, batch_size = 1)
        gen_data.write_prediction_vol(k, im_pred)
        #results[k,...]=im_pred
        #pdb.set_trace()
        
    t1=time.time()
    
    print "prediction of ({0}, {1}, {2}) took {3} seconds".format(patchZ_out, patchSize_out, patchSize_out, (time.time() - t0))
    #pdb.set_trace()   
    gen_data.save_to_disk(savename)
    

