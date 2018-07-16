#! /usr/bin/env python

volume_name = np.array(['A','B','C','A+','B+','C+'])
bad_slices = np.array([[143],[1,29,30,58,59,91],[28,88,100],
                       [65,93,94,122,123,125],[1,29,30,58,59,91],[28,88,100]])
#to be replaced? same name with bad_slice??
bb = np.array([[143],[1,29,30,58,59,91],[28,88,100]])
gg = np.array([[142],[0,28,31,57,60,90],[27,87,99],
               [64,92,95,121,124,126],[0,28,31,57,60,90],[27,87,99]])
#newly-aligned image size
new_align_sz = np.array([[1727,1842],[2069,1748],[1986,2036],
                         [1741,1912],[2898,1937],[1914,1983]])
# CREMI ground truth : 125,250,1250
# _v2_200: 200 margin from manual label


def origin2align(option='raw',nn=volume_name):
    '''
        orig -> align_v2_200 (translation)
        gray image(raw)
        seg/syn
        option:raw,syn
        '''
    if option=='raw':
        output = {}
        for nid in range(nn.shape[0]):
            vol = nn[nid]
            if len(vol)==2:
                sn='06'
            else:
                sn = '05'
            if vol =='A':
                oo=24
            else:
                oo=23
            pw=0
            ph=0
            if vol[0]=='B':
                ph=200
            elif len(vol)==2:
                ph=700
            #print ('data/public/sample_'+vol+'_padded_2016'+sn+'01.hdf')
            pp = np.cumsum(np.loadtxt('data/align/trans_'+vol+'_v2.txt',delimiter=','),axis=0)
            pp = pp[76] - pp
            #ww=200;suf='v2'
            ww = np.ceil([np.max(pp[:,0]),np.max(pp[:,1]),-np.min(pp[:,0]),-np.min(pp[:,1])]).astype('int')+200
            #print (ww)
            im = h5py.File('data/public/sample_'+vol+'_padded_2016'+sn+'01.hdf')['volumes/raw'][:]
            #im = tmphdf5 #!
            out = np.zeros([153,1250+np.sum(ww[np.array([1,3])]).astype('int'),1250+np.sum(ww[np.array([0,2])]).astype('int')],dtype='uint8')
            suf='v2_200'
            for i in range(153):
                pd = np.round(pp[i,:]).astype('int')
                im2 = np.pad(im[oo+i,:,:],(ph,pw),'symmetric')
                #print (out.shape)
                out[i,:,:] = im2[(912+pd[1]-ww[1]+pw-1):(911+pd[1]+1250+ww[3]+pw),
                                 (912+pd[0]-ww[0]+ph-1):(911+pd[0]+1250+ww[2]+ph)]
            out[bad_slices[nid],:,:]=out[gg[nid],:,:]  #in matlab write gg{nid}+1, attention
            output[nid] = out
        return output
    elif option=='syn':
        output_syn = {}
        output_seg = {}
        for nid in range(3):
            vol = nn[nid]
            if len(vol)==2:
                sn='06'
            else:
                sn = '05'
            ph=0
            if vol[0]=='B':
                ph=200
            elif len(vol)==2:
                ph=700
            if vol =='A':
                oo=24
            else:
                oo=23
            syn = h5py.File('data/public/sample_'+vol+'_padded_2016'+sn+'01.hdf')['volumes/labels/clefts'][:]
            syn[syn>=1e10] = 0
            seg = h5py.File('data/public/sample_'+vol+'_padded_2016'+sn+'01.hdf')['volumes/labels/neuron_ids'][:]
            pp = np.cumsum(np.loadtxt('data/align/trans_'+vol+'_v2.txt',delimiter=','),axis=0)
            pp = pp[76] - pp
            ww = np.ceil([np.max(pp[:,0]),np.max(pp[:,1]),-np.min(pp[:,0]),-np.min(pp[:,1])]).astype('int')+200
            seg_o = np.zeros([153,1250+np.sum(ww[np.array([0,2])]).astype('int'),1250+np.sum(ww[np.array([1,3])]).astype('int')],dtype='uint64')
            syn_o = np.zeros([153,1250+np.sum(ww[np.array([0,2])]).astype('int'),1250+np.sum(ww[np.array([1,3])]).astype('int')],dtype='uint16')
            for i in range(125):
                pd = np.round(pp[i+14-1,:]).astype('int')
                tmp = np.zeros([3075+ph,3075])
                tmp[912-1:911+1250,912-1:911+1250] = seg[i,:,:]
                seg_o[i+14-1,:,:] = tmp[(912+pd[0]-ww[0]-1):(911+pd[0]+1250+ww[2]),
                                        (912+pd[1]-ww[1]-1):(911+pd[1]+1250+ww[3])]
                                        tmp[912-1:911+1250,912-1:911+1250] = seg[i,:,:]
                                        syn_o[i+14-1,:,:] = tmp[(912+pd[0]-ww[0]-1):(911+pd[0]+1250+ww[2]),
                                                                (912+pd[1]-ww[1]-1):(911+pd[1]+1250+ww[3])]
            seg_o[bad_slices[nid],:,:]=seg_o[gg[nid],:,:]
            syn_o[bad_slices[nid],:,:]=syn_o[gg[nid],:,:]
            output_syn[nid] = syn_o
            output_seg[nid] = seg_o
        return output_syn,output_seg


def align2origin(option='raw',nn=volume_name,sz = new_align_sz):
    for nid in range(6):
        vol = nn[nid]
        if len(vol)==2:
            sn='06'
        else:
            sn = '05'
        % load/crop result
        result = h5py.File('data/cremi/images/im_'+vol+'_v2_200.h5')['main'][:]
        sz_r = result.shape
        sz_bd = np.round((sz_r - np.array([125,sz[i][0]-400,sz[i][0]-400]))/2.).astype('int')
        result = result[sz_bd[0]:-sz_bd[0], sz_bd[1]:-sz_bd[1], sz_bd[2]:-sz_bd[2]];

        pp = np.cumsum(np.loadtxt('data/align/trans_'+vol+'_v2.txt',delimiter=','),axis=0)
        pp = pp[76] - pp
        ww = np.ceil([np.max(pp[:,0]),np.max(pp[:,1]),-np.min(pp[:,0]),-np.min(pp[:,1])]).astype('int')
        pp=pp[14:-14,:]
        # 1250+200*2
        result_o = np.zeros([125,1250,1250],'uint16');
        for i in range(125):
            pd = np.round(pp[i+14-1,:]).astype('int')

            result_o[i,:,:] = result[(pd[0]+ww[0]-1):(pd[0]+1250+ww[0]),
                                     (pd[1]+ww[1]-1):(pd[1]+1250+ww[1])]

