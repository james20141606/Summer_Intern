% volume names
nn={'A','B','C','A+','B+','C+'};
% bad slices 
bb={[143],[1,29,30,58,59,91],[28,88,100],...
    [65,93,94,122,123,125],[1,29,30,58,59,91],[28,88,100]};
% to be replaced
gg={[142],[0,28,31,57,60,90],[27,87,99],...
    [64,92,95,121,124,126],[0,28,31,57,60,90],[27,87,99]};
% newly-aligned image size
sz={[1727,1842],[2069,1748],[1986,2036],[1741,1912],[2898,1937],[1914,1983]};

tid=1

% CREMI: 125,250,1250
% _v2_200: 200 margin from manual label
switch floor(tid)
case 1 % orig -> align_v2_200 (translation)
    switch tid
    case 1 % gray image
        for nid=1:numel(nn)
            
            vol = nn{nid}
            disp(vol)
            sn='05';if numel(vol)==2;sn='06';end
            oo=23; if strcmp(vol,'A');oo=24;end
            pw=0;ph=0; if vol(1)=='B';ph=200;if numel(vol)==2;ph=700;end;end 
            disp(['cremi/images/sample_' vol '_padded_2016' sn '01.hdf'])
            im = h5read(['cremi/images/sample_' vol '_padded_2016' sn '01.hdf'],'/volumes/raw');
            pp=cumsum(load(['cremi/align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            %ww=200;suf='v2';
            ww = ceil([max(pp) -min(pp)])+200;suf='v2_200';
            out=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153]);%,'uint8');
            for i=1:153
                pd = round(pp(i,:)); 
                im2 = padarray(im(:,:,oo+i),[ph,pw],'symmetric','both');
                out(:,:,i) = im2((912+pd(1)-ww(1)+ph):(911+pd(1)+1250+ww(3)+ph),...
                                (912+pd(2)-ww(2)+pw):(911+pd(2)+1250+ww(4)+pw)); 
            end
            out(:,:,bb{nid}+1)=out(:,:,gg{nid}+1);
            
            %disp(unique(out-im))
            h5create(['cremi/images/im_' sprintf('%s_%s.h5',vol,suf)],'/main',size(out));
            h5write(['cremi/images/im_' sprintf('%s_%s.h5',vol,suf)],'/main',out);
        end
    case 1.1 % seg/syn
        for nid=1:3
            vol = nn{nid}
            disp(vol)
            sn='05';if numel(vol)==2;sn='06';end
            oo=23; if strcmp(vol,'A');oo=24;end
            syn = h5read([D0 'images/sample_' vol '.hdf'],'/volumes/labels/clefts');
            syn(syn>1e10)=0;
            seg = h5read([D0 'images/sample_' vol '.hdf'],'/volumes/labels/neuron_ids');
            pp=cumsum(load(['align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            % 1250+200*2
            %ww=200;suf='v2';
            ww = ceil([max(pp) -min(pp)])+200;suf='v2_200';
            seg_o=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153],'uint64');
            syn_o=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153],'uint16');
            for i=1:125
                pd = round(pp(i+14,:)); 
                tmp = zeros(3075+ph,3075);

                tmp(912:911+1250,912:911+1250) = seg(:,:,i);
                seg_o(:,:,i+14) = tmp((912+pd(1)-ww(1)):(911+pd(1)+1250+ww(3)),...
                                (912+pd(2)-ww(2)):(911+pd(2)+1250+ww(4))); 
                tmp(912:911+1250,912:911+1250) = syn(:,:,i);
                syn_o(:,:,i+14) = tmp((912+pd(1)-ww(1)):(911+pd(1)+1250+ww(3)),...
                                (912+pd(2)-ww(2)):(911+pd(2)+1250+ww(4))); 
            end
            seg_o(:,:,bb{nid}+1)=seg_o(:,:,gg{nid}+1);
            syn_o(:,:,bb{nid}+1)=syn_o(:,:,gg{nid}+1);
            U_h5write(['../data/cremi/align_v2/seg_' sprintf('%s_%s.h5',vol,suf)],'/main',seg_o,5,'uint64');
            U_h5write(['../data/cremi/align_v2/syn_' sprintf('%s_%s.h5',vol,suf)],'/main',syn_o,5,'uint16');
        end
    end
case 2 % align_v2_200 (translation) -> orig
    switch tid
    case 2.1 % seg/syn
        for nid=1:6
            vol = nn{nid}
            sn='05';if numel(vol)==2;sn='06';end
            % load/crop result
            result = h5read(['results/sample_' vol '.hdf'],'/main');
            sz_r = size(result);
            sz_bd = round((sz_r-[sz{nid}-400 125])/2);
            result = result(sz_bd(1)+1:end-sz_bd(1), sz_bd(2)+1:end-sz_bd(2), sz_bd(3)+1:end-sz_bd(3)); 

            pp=cumsum(load(['align/trans_' vol '_v2.txt']),1);
            pp=bsxfun(@minus,pp,pp(77,:));
            ww = ceil([max(pp) -min(pp)]);
            pp=pp(15:end-14,:);
            % 1250+200*2
            result_o = zeros([1250,1250,125],'uint16');
            for i=1:125
                pd = round(pp(i+14,:)); 

                result_o(:,:,i) = result((pd(1)+ww(1)):(pd(1)+1250+ww(1)),...
                                         (pd(2)+ww(2)):(pd(2)+1250+ww(2))); 
            end
            %U_h5write(['../data/cremi/align_v2/syn_' sprintf('%s_%s.h5',vol,suf)],'/main',syn_o,5,'uint16');
        end
    end
end
