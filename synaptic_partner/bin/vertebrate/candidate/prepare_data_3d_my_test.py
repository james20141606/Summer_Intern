import time
import glob
#import mahotas
import numpy as np
import scipy
import scipy.misc
import random
from keras.models import Model, Sequential, model_from_json
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter, label, find_objects, distance_transform_edt
import pickle
from rotate3d import *
import itertools
import pdb
from scipy.spatial import KDTree
import os

def relabel_from_one(a):
    labels = np.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return a, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0)+1)
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map



class GenerateData:
    
    def __init__(self, trial_name, datadir, imagedir, predname, gtname=None, segname=None, patchSize=250, patchZ=24, margin=False, ignore=False):
        #pdb.set_trace()    
        if gtname!=None:
            fid = h5py.File(os.path.join(datadir,gtname))
            gtvol = np.array(fid['stack'])
            fid.close()
            
            self.gtvol=gtvol.astype(np.int32)
        
        fid = h5py.File(os.path.join(datadir,predname))
        self.predvol = np.array(fid['stack'])
        fid.close()
        #self.predvol = self.predvol[14:-14,44:-44,44:-44]
        
        if margin:
            self.predvol[:14,...]=0.0
            self.predvol[-15:,...]=0.0
        if ignore:
            self.gtvol[self.gtvol>200]=0 #not sure ones
        
        fid = h5py.File(os.path.join(datadir,segname))
        self.segvol = np.array(fid['stack'])
        fid.close()
        
        allfiles = sorted(glob.glob(os.path.join(datadir,imagedir+'/*.png')))
        #pathPrefix = datadir
        #img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_labelImages    = pathPrefix + 'labels/' + purpose + '/'+block_name+'/*.tif'
        #img_search_string_grayImages     = pathPrefix + 'images/' + purpose + '/'+block_name+'/*.tif'

        
        #img_files_gray     = sorted( glob.glob( img_search_string_grayImages ) )
        #img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )
        #img_files_labels   = sorted( glob.glob( img_search_string_labelImages ) )


        img = np.array(Image.open(allfiles[0])) #read the first image to get imformation about the shape
        
        self.grayImages    = np.zeros((len(allfiles), img.shape[0], img.shape[1])).astype(np.float32)
        


        #read_order = range(np.shape(img_files_gray)[0])
        #for img_index in read_order:
        for ii,filename in enumerate(allfiles):
            #print filename
            img = np.array(Image.open(filename)).astype(np.float32)
            
            img_normalize_toufiq = img *1.0/255

            self.grayImages[ii,:,:] = img_normalize_toufiq
        
        #pdb.set_trace()
        
        self.patchSize = patchSize
        self.patchZ = patchZ
        self.trial_name = trial_name
        self.datadir = datadir
        
        if not os.path.exists(self.trial_name):
            os.makedirs(self.trial_name)
        
        
        
        self.cc_partners = pickle.load(open(os.path.join(self.trial_name,'saved_cc_partners_with_label.pkl'),'rb'))
        self.cc_partner_keys = self.cc_partners.keys()
        
    def inside_margin(self, contact_loc):
        
        depth, height, width = self.segvol.shape
        
        zedge=0
        if contact_loc[0] < self.patchZ/2 or contact_loc[0]> (depth-(self.patchZ/2)):
            zedge=1
        yedge=0    
        if contact_loc[1] < self.patchSize/2 or contact_loc[1]> (height-(self.patchSize/2)):
            yedge = 1
        xedge = 0
        if contact_loc[2] < self.patchSize/2 or contact_loc[2]> (width-(self.patchSize/2)):
            xedge = 1
        
        sum_edge = yedge +xedge
        
        if zedge>0 or sum_edge>0:
            return False
        else:
            return True
        
    def inside_margin2(self, contact_loc):
        
        depth, height, width = self.segvol.shape
        
        flag = (contact_loc[:,0] >= (self.patchZ/2)) * (contact_loc[:,0] < (depth-(self.patchZ/2))) * (contact_loc[:,1] >= (self.patchSize/2)) * (contact_loc[:,1] < (height-(self.patchSize/2))) * (contact_loc[:,2] >= (self.patchSize/2)) * (contact_loc[:,2] < (width-(self.patchSize/2)))
        
        flag= np.array(flag).astype(int)
        
        if np.sum(flag)>0:
            found=True
            idx=np.where(flag==1)
        else:
            found=False
            idx=-1
        return found, idx
        
    def ntrue_positives_inside_margin(self, only_cleft=False):
	#pdb.set_trace()
	gtvol = self.gtvol[self.patchZ/2:-(self.patchZ/2),self.patchSize/2:-self.patchSize/2,self.patchSize/2:-self.patchSize/2]
        
	if only_cleft==True:
	    uids_common=np.setdiff1d(np.unique(gtvol),[0])
	
	else:
	    pre_mask= ((gtvol%2)==1)
            post_mask= ((gtvol%2)==0)
            uids_pre = np.setdiff1d(np.unique(gtvol[pre_mask]),[0])
            uids_post = np.setdiff1d(np.unique(gtvol[post_mask]),[0])
            uids_common = set(uids_pre+1).intersection(uids_post)
	return len(uids_common)

    def get_num_candidates(self):
        
        self.candidates=[]
        count=0
        positive_ids=[]
        for sid in self.cc_partner_keys:
            set1 = self.cc_partners[sid]
            for jj in range(len(set1)):
                found, idx = self.inside_margin2(self.cc_partners[sid][jj]['contact_loc'])
                if found:
                    contact_ctr = self.cc_partners[sid][jj]['contact_loc'][idx[0][0]]
                    self.cc_partners[sid][jj]['contact_loc_ctr'] = contact_ctr
                    self.candidates.append(self.cc_partners[sid][jj])
                    count = count + 1
                    if self.cc_partners[sid][jj]['label']==1:
                        positive_ids.append(self.cc_partners[sid][jj]['gt_id'])
                    
        print '# positive ids: ',len(np.unique(positive_ids))
        #pdb.set_trace()
	return count
            
            
            
        
    def compute_cube(self,partner_entry):
        #pdb.set_trace()
        patchZ = self.patchZ
        patchSize = self.patchSize
        patchZ_half = patchZ/2
        patchSize_half = patchSize/2
        count=0
        #for uid in self.cc_partners.keys():
            #for ci in range(len(self.cc_partners[uid])):
        post_id = partner_entry['post']
        post_seg = partner_entry['post_seg']
        pre_id = partner_entry['pre']
        pre_seg = partner_entry['pre_seg']
        
        #print '{0} -> {1} : seg {2} -> {3}'.format(pre_id,post_id, pre_seg, post_seg)
        #random_locid = random.sample(range(len(partner_entry['contact_loc'])),1)[0]
        contact_loc = partner_entry['contact_loc_ctr']
        
        
        zmin = max(0,contact_loc[0]-patchZ_half)
        zmax = min(contact_loc[0]+patchZ_half,self.predvol.shape[0])
        ymin = max(0,contact_loc[1]-patchSize_half)
        ymax = min(contact_loc[1]+patchSize_half,self.predvol.shape[1])
        xmin = max(0,contact_loc[2]-patchSize_half)
        xmax = min(contact_loc[2]+patchSize_half,self.predvol.shape[2])
        
        bzmin = max(0,patchZ_half-contact_loc[0])
        bzmax = patchZ - max(0, (contact_loc[0]+patchZ_half)-self.predvol.shape[0])
        bymin = max(0,patchSize_half-contact_loc[1])
        bymax = patchSize - max(0, (contact_loc[1]+patchSize_half)-self.predvol.shape[1])
        bxmin = max(0,patchSize_half-contact_loc[2])
        bxmax = patchSize - max(0, (contact_loc[2]+patchSize_half)-self.predvol.shape[2])
        
        predvol = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        predvol[bzmin:bzmax, bymin:bymax, bxmin:bxmax] =  self.predvol[zmin:zmax, ymin:ymax, xmin:xmax]
        
        segvol_pre = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        segvol_pre[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.segvol[zmin:zmax, ymin:ymax, xmin:xmax]
        segvol_pre = (segvol_pre==pre_seg).astype(np.float32)
        
        segvol_post = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        segvol_post[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.segvol[zmin:zmax, ymin:ymax, xmin:xmax]
        segvol_post = (segvol_post==post_seg).astype(np.float32)

        imgvol = np.zeros((patchZ,patchSize,patchSize)).astype(np.float32)
        imgvol[bzmin:bzmax, bymin:bymax, bxmin:bxmax] = self.grayImages[zmin:zmax, ymin:ymax, xmin:xmax]
                
                #count=count+1
                #fid = h5py.File('samples/sample_'+str(count).zfill(5)+'_label'+str(self.cc_partners[post_id][ci]['label'])+'.h5','w')
                #fid.create_dataset('imgvol',data=imgvol)
                #fid.create_dataset('predvol',data=predvol)
                #fid.create_dataset('segvol_pre',data=segvol_pre)
                #fid.create_dataset('segvol_post',data=segvol_post)
                #fid.close()
                #pdb.set_trace()
        return imgvol, predvol, segvol_pre,segvol_post
    
    def get_next_test_sample(self, sample_id):

        combined0 = self.compute_cube(self.candidates[sample_id])
        #pdb.set_trace()     
        transformed = []
        transformed.append([])
        transformed[-1] = combined0
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):
                        #for nrotate in range(4):
                        combined = self.reflect_swap(combined0, reflectz, reflecty, reflectx, swapxy)
                        #combined = self.rotate3d(combined, nrotate)
                        transformed.append([])
                        transformed[-1]=combined

        #combined = self.compute_cube(self.candidates[sample_id])
        
        label = self.candidates[sample_id]['label']

        pre = self.candidates[sample_id]['pre']
        post = self.candidates[sample_id]['post']
        if self.candidates[sample_id].has_key('gt_id'):
            gt_id = self.candidates[sample_id]['gt_id']
        else: 
            gt_id = 0
        #data_set = (combined, label, pre, post, gt_id)
        data_set = (transformed, label, pre, post, gt_id)
        


        return data_set

    def store_prediction(self,sample_id, pred):
   
       self.candidates[sample_id]['pred'] = pred 
     
    def save_candidates_with_prediction(self):

       savename = os.path.join(self.trial_name,'saved_cc_partners_with_predictions.pkl')
       pickle.dump(self.candidates, open(savename,'wb'))


    def get_3d_sample2(self, nsamples_batch=5 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        npositive = 5
        nnegative = nsamples-npositive
        
        
        input_set = np.zeros((nsamples, 4, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        random_pidx = random.sample(range(len(self.positive_examples)), npositive)
        random_nidx = random.sample(range(len(self.positive_examples)), nnegative)


        count=0
        for pidx in random_pidx:
            pentry = self.positive_examples[pidx]
            combined = self.compute_cube(pentry)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
        for nidx in random_nidx:
            pentry = self.negative_examples[nidx]
            combined = self.compute_cube(pentry)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
            
    
        #start_time = time.time()
        
        

        print "number of positive labels: "+str(np.sum(label_set>0))


        data_set = (input_set, label_set)
        

        #end_time = time.time()
        #total_time = (end_time - start_time)


        return data_set

    def rotate3d(self, input_vol, nrotate):

        #pdb.set_trace()
        #nrotate = random.randint(0,3)
        axis = 0 #random.randint(0,2)

        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            grayImg_set = axial_rotations(grayImg_set, rot=nrotate, ax=axis)

            output_vol.append([])
            output_vol[-1] = grayImg_set

        return output_vol

    def reflect_swap(self, input_vol, reflectz, reflecty, reflectx, swapxy):


        #reflectz=random.randint(0,1)
        #reflecty=random.randint(0,1)
        #reflectx=random.randint(0,1)
        #swapxy=random.randint(0,1)

        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            if reflectz:
                grayImg_set = grayImg_set[::-1,:,:]

            if reflecty:
                grayImg_set = grayImg_set[:,::-1,:]

            if reflectx:
                grayImg_set = grayImg_set[:,:,::-1]

            if swapxy:
                grayImg_set = grayImg_set.transpose((0,2,1))

            output_vol.append([])
            output_vol[-1] = grayImg_set

        return output_vol
    
    
        
    def scale_intensity(self, input_vol):
            
        grayImg_set = input_vol
            
        self.scale_low = 0.8
        self.scale_high = 1.2
        self.shift_low = -0.2
        self.shift_high = 0.2


        grayImg_set_mean = grayImg_set.mean()
        grayImg_set = grayImg_set_mean + (grayImg_set -grayImg_set_mean)*np.random.uniform(low=self.scale_low,high=self.scale_high)
        grayImg_set = grayImg_set + np.random.uniform(low=self.shift_low,high=self.shift_high)
        grayImg_set = np.clip(grayImg_set, 0.05, 0.95)
        
        return grayImg_set
    

