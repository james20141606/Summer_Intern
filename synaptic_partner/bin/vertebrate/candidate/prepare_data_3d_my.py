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
            print filename
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
        
        self.separate_class()


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
        
        if zedge>0 or sum_edge>1:
            return False
        else:
            return True
        
    def inside_margin2(self, contact_loc):
        
        depth, height, width = self.segvol.shape
        
        flag = (contact_loc[:,0] > self.patchZ/2) * (contact_loc[:,0] < (depth-(self.patchZ/2))) * (contact_loc[:,1] > self.patchSize/2) * (contact_loc[:,1] < (height-(self.patchSize/2))) * (contact_loc[:,2] > self.patchSize/2) * (contact_loc[:,2] < (width-(self.patchSize/2)))
        
        flag= np.array(flag).astype(int)
        
        if np.sum(flag)>0:
            found=True
            idx=np.where(flag==1)[0]
            #pdb.set_trace()
            maxlen = min(5, len(idx))
            ridx = random.sample(idx[:maxlen],1)[0]
        else:
            found=False
            ridx=-1
        return found, ridx
        
    def separate_class(self):
        
        #pdb.set_trace()
        all_locations=[]
        loc_map={}
        self.positive_examples=[]
        self.negative_examples=[]
        for sid in self.cc_partner_keys:
            set1 = self.cc_partners[sid]
            for pid, entry in enumerate(set1):
                found,lidx = self.inside_margin2(entry['contact_loc'])
                if found==True:
                    if entry['label']==1:
                        self.positive_examples.append(entry)
                        #all_locations.append(entry['contact_loc'][5])
                        #loc_map[len(all_locations)-1] = [0, len(self.positive_examples)-1]
                    elif entry['label']==-1:
                        self.negative_examples.append(entry)
                        #all_locations.append(entry['contact_loc'][5])
                        #loc_map[len(all_locations)-1] = [1, len(self.negative_examples)-1]
                    
        print "total positive {0}".format(len(self.positive_examples))    
        print "total negative {0}".format(len(self.negative_examples))    
        #all_locations = np.array(all_locations)
        #all_locations[:,0] = all_locations[:,0]*30
        #all_locations[:,1] = all_locations[:,1]*4
        #all_locations[:,1] = all_locations[:,2]*4
        
        #from scipy.spatial.distance import pdist, squareform
        #dist=pdist(all_locations,'euclidean')
        #dist_matrix=squareform(dist)
        
        #rows=(dist_matrix<10).nonzero()[0]
        #cols=(dist_matrix<10).nonzero()[1]
        #pdb.set_trace()
        
        
    def compute_cube(self,partner_entry):
        #pdb.set_trace()
        patchZ = self.patchZ
        patchSize = self.patchSize
        patchZ_half = patchZ/2
        patchSize_half = patchSize/2
        count=0
        depth,height,width=self.segvol.shape
        #for uid in self.cc_partners.keys():
            #for ci in range(len(self.cc_partners[uid])):
        post_id = partner_entry['post']
        post_seg = partner_entry['post_seg']
        pre_id = partner_entry['pre']
        pre_seg = partner_entry['pre_seg']
        #pdb.set_trace()
        found, random_locid = self.inside_margin2(partner_entry['contact_loc'])
        contact_loc0 = partner_entry['contact_loc'][random_locid]
        
        #z_disp = random.sample([-2,0,2], 1)[0]
        #y_disp = random.sample([-10,0,10], 1)[0]
        #x_disp = random.sample([-10,0,10], 1)[0]
        z_disp = random.sample(range(-2,2), 1)[0]
        y_disp = random.sample(range(-10,10), 1)[0]
        x_disp = random.sample(range(-10,10), 1)[0]
        
        contact_loc = contact_loc0 + np.array([z_disp, y_disp, x_disp])
        if contact_loc[0] < patchZ_half: contact_loc[0] = contact_loc0[0]
        if contact_loc[1] < patchSize_half: contact_loc[1] = contact_loc0[1]
        if contact_loc[2] < patchSize_half: contact_loc[2] = contact_loc0[2]
        
        if contact_loc[0] > (depth+patchZ_half): contact_loc[0] = contact_loc0[0]
        if contact_loc[1] > (height+patchSize_half): contact_loc[1] = contact_loc0[1]
        if contact_loc[2] > (width+patchSize_half): contact_loc[2] = contact_loc0[2]
        
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
    
    def get_3d_sample(self, nsamples_batch=5 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        
        random_idx = random.sample(self.cc_partner_keys, nsamples)
        
        input_set = np.zeros((nsamples, 4, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        for i,idx in enumerate(random_idx):
            npt = len(self.cc_partners[idx])
            random_idx_j = random.sample(range(npt),1)[0]
            
            combined = self.compute_cube(idx, random_idx_j)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[i,:,:,:,:] = combined
            label_set[i] = self.cc_partners[idx][random_idx_j]['label']
            
            
    
        #start_time = time.time()
        
        

        print "number of positive labels: "+str(np.sum(label_set>0))


        data_set = (input_set, label_set)
        

        #end_time = time.time()
        #total_time = (end_time - start_time)


        return data_set

    def get_3d_sample2(self, nsamples_batch=5 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        npositive = 5
        nnegative = nsamples-npositive
        
        
        input_set = np.zeros((nsamples, 4, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        random_pidx = random.sample(range(len(self.positive_examples)), npositive)
        random_nidx = random.sample(range(len(self.negative_examples)), nnegative)

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

    def rotate3d(self, input_vol):

        #pdb.set_trace()
        nrotate = random.randint(0,3)
        axis = 0 #random.randint(0,2)
        
        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            grayImg_set = axial_rotations(grayImg_set, rot=nrotate, ax=axis)
        
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    def reflect_swap(self, input_vol):

        
        reflectz=random.randint(0,1)
        reflecty=random.randint(0,1)
        reflectx=random.randint(0,1)
        swapxy=random.randint(0,1)

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
    

    #def get_test_sample(self, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        
 	#pdb.set_trace()       
        
        #rangex1 = set(range(self.patchSize/2, self.grayImages.shape[2]-self.patchSize/2, patchSize_out))
        #rangex1.add(self.grayImages.shape[2]-patchSize/2)
        #rangex = sorted(list(rangex1))

        #rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        #rangey1.add(self.grayImages.shape[1]-patchSize/2)
        #rangey = sorted(list(rangey1))
        
        #rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        #rangez1.add(self.grayImages.shape[0]-patchZ/2)
        #rangez = sorted(list(rangez1))

	#nsamples = len(rangez)*len(rangey)*len(rangex) 
	#grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
       
        #count=0
        #for z_index in rangez:
            #for y_index in rangey:
                #for x_index in  rangex:
                    #grayImg_set[count,:,:,:] = self.grayImages[z_index-(patchZ/2):z_index+(patchZ/2), y_index-(patchSize/2):y_index+(patchSize/2), x_index-(patchSize/2):x_index+(patchSize/2)]
                    
                    #count =count+1
                    
        #grayImg_set = grayImg_set[:count,...]
        #return grayImg_set
                    
    #def compute_3d_result(self, prediction1D, savename, nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=7, patchZ_out=1, doAugmentation=False):

        #cropSize = (patchSize - patchSize_out)/2
        #csZ = (patchZ - patchZ_out)/2


        #result_vol = np.zeros(self.grayImages.shape)

        #count=0
        
        #rangex1 = set(range(patchSize/2, self.grayImages.shape[2]-patchSize/2, patchSize_out))
        #rangex1.add(self.grayImages.shape[2]-patchSize/2)
        #rangex = sorted(list(rangex1))

        #rangey1 = set(range(patchSize/2, self.grayImages.shape[1]-patchSize/2, patchSize_out))
        #rangey1.add(self.grayImages.shape[1]-patchSize/2)
        #rangey = sorted(list(rangey1))
        
        #rangez1 = set(range(patchZ/2, self.grayImages.shape[0]-patchZ/2, patchZ_out))
        #rangez1.add(self.grayImages.shape[0]-patchZ/2)
        #rangez = sorted(list(rangez1))
        
        #for z_index in rangez:
            #for y_index in rangey:
                #for x_index in  rangex:
                    
                    
                    #result_reshaped = prediction1D[count,...] 
                    #result_reshaped = result_reshaped.reshape((patchZ_out, patchSize_out, patchSize_out))
                    
                    #lbl_startz = z_index-(patchZ/2)+csZ
                    #lbl_endz = lbl_startz +patchZ_out
                    #lbl_starty = y_index-(patchSize/2)+cropSize
                    #lbl_endy = lbl_starty + patchSize_out
                    #lbl_startx = x_index-(patchSize/2)+cropSize
                    #lbl_endx = lbl_startx+patchSize_out
                    
                    #result_vol[lbl_startz:lbl_endz, lbl_starty:lbl_endy, lbl_startx:lbl_endx] =  result_reshaped
                    
                    #count =count+1
                    
        #fidw=h5py.File(savename,'w')
        #fidw.create_dataset('stack',data=result_vol.astype(np.float32))
        #fidw.close()

