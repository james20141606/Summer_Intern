import sys
import numpy as np
DD='/mnt/coxfs01/donglai/'
DD2='/var/www/data/cerebellum/'

opt=sys.argv[1]
if opt[0]=='0':
    # copy pred data from coxfs
    import shutil
    pref=''
    if opt =='0': # from prediction
        Did=DD+'data/cerebellum/data/syn0619_proof/';
        #sn = 'fp'
        sn = 'tp'
        num = np.loadtxt(DD+sn+'_id.txt',delimiter=',').astype(int)
        D0=DD+'cerebellum/db/syn0619/manual_200_v2/'
    elif opt =='0.1': # from gt
        Did=DD+'cerebellum/db/syn0619/seg_bbox/pred_0619/result_200_v2_v2_check.txt'
        sn = 'tp'
        bb = np.loadtxt(Did,delimiter=',').astype(int)
        num = np.where(bb>=0)[0]
        D0=DD+'cerebellum/db/syn0619/gt_200_v2/'
        pref='gt_'
    D1='/home/donglai/data/cerebellum/roi_label/'+sn+'/'
    for nn in num:
        shutil.copytree(D0+str(nn),D1+pref+str(nn))

elif opt[0]=='1': # data id
    if opt == '1': # gt-fn
        bb = np.loadtxt(DD+'cerebellum/db/syn0619/seg_bbox/pred_0619/result_200_v2_v2_check.txt',delimiter=',').astype(int)
        fn = np.where(bb<0)[0]
        out = ''
        for o in fn:
            out+='gt_'+o+','
        b= open(DD2+'roi_label/data/fn_id.txt','w')
        b.write(out[:-1])
        b.close()
    elif opt == '1.01': # pred+gt - tp
        out = ''
        # within gt
        bb = np.loadtxt(DD+'cerebellum/db/syn0619/seg_bbox/pred_0619/result_200_v2_v2_check.txt',delimiter=',').astype(int)
        for b in np.where(bb>-1)[0]:
            out+='gt_'+str(b)+','
        # within pred
        cc = np.loadtxt(DD+'data/cerebellum/data/syn0619_proof/tp_id.txt',delimiter=',').astype(int)
        for c in cc:
            out+=str(c)+','
        b = open(DD2+'roi_label/data/tp_id.txt','w')
        b.write(out[:-1])
        b.close()
    elif opt == '1.02': # pred-fp
        num = np.loadtxt(DD+'data/cerebellum/data/syn0619_proof/fp_id.txt',delimiter=',').astype(int)
        out = ''
        for o in num:
            out+=str(o)+','
        b= open(DD2+'roi_label/data/fp_id.txt','w')
        b.write(out[:-1])
        b.close()
