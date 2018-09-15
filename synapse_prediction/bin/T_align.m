D0='/n/coxfs01/vcg_connectomics/cremi/';
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
tid=2.1
% CREMI: 125,250,1250
% _v2_200: 200 margin from manual label
switch floor(tid)
case 1 % orig -> align_v2_200 (translation)
    switch tid
    case 1 % gray image
        for nid=1:numel(nn)
            vol = nn{nid};
            sn='05';if numel(vol)==2;sn='06';end
            oo=23; if strcmp(vol,'A');oo=24;end
            pw=0;ph=0; if strcmp(vol,'B+');ph=700;end
            im = h5read([D0 'images/orig/sample_' vol '_padded_2016' sn '01.hdf'],'/volumes/raw');
            
            pp=cumsum(load([D0 'align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            ww = ceil([max(pp) -min(pp)])+200;suf='v2_200';
            out=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153],'uint8');
            for i=1:153
                pd = round(pp(i,:)); 
                im2 = padarray(im(:,:,oo+i),[ph,pw],'symmetric','both');
                out(:,:,i) = im2((912+pd(1)-ww(1)+ph):(911+pd(1)+1250+ww(3)+ph),...
                                (912+pd(2)-ww(2)+pw):(911+pd(2)+1250+ww(4)+pw)); 
            end
            out(:,:,bb{nid}+1)=out(:,:,gg{nid}+1);
            %U_h5write(['images/im_' sprintf('%s_%s.h5',vol,suf)],'/main',out);
        end
    case 1.1 % seg/syn
        for nid=2%1:3
            vol = nn{nid}
            sn='05';if numel(vol)==2;sn='06';end

            syn = h5read([D0 'images/orig/sample_' vol '_2016' sn '01.hdf'],'/volumes/labels/clefts');
            syn(syn>1e10)=0;
            seg = h5read([D0 'images/orig/sample_' vol '_2016' sn '01.hdf'],'/volumes/labels/neuron_ids');

            pp=cumsum(load([D0 'align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            % 1250+200*2
            ww = ceil([max(pp) -min(pp)])+200;suf='v2_200';
            seg_o=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153],'uint64');
            syn_o=zeros([1250+sum(ww([1,3])),1250+sum(ww([2,4])),153],'uint16');
            for i=1:125
                pd = round(pp(i+14,:)); 
                tmp = zeros(3072,3072);

                tmp(912:911+1250,912:911+1250) = seg(:,:,i);
                seg_o(:,:,i+14) = tmp((912+pd(1)-ww(1)):(911+pd(1)+1250+ww(3)),...
                                (912+pd(2)-ww(2)):(911+pd(2)+1250+ww(4))); 
                tmp(912:911+1250,912:911+1250) = syn(:,:,i);
                syn_o(:,:,i+14) = tmp((912+pd(1)-ww(1)):(911+pd(1)+1250+ww(3)),...
                                (912+pd(2)-ww(2)):(911+pd(2)+1250+ww(4))); 
            end
            seg_o(:,:,bb{nid}+1)=seg_o(:,:,gg{nid}+1);
            syn_o(:,:,bb{nid}+1)=syn_o(:,:,gg{nid}+1);
            %U_h5write(['../data/cremi/align_v2/seg_' sprintf('%s_%s.h5',vol,suf)],'/main',seg_o,5,'uint64');
            %U_h5write(['../data/cremi/align_v2/syn_' sprintf('%s_%s.h5',vol,suf)],'/main',syn_o,5,'uint16');
        end
    end
case 2 % align_v2_200 (translation) -> orig
    switch tid
    case 2.1 % seg/syn
        for nid=4:6
            vol = nn{nid}
            sn='05';if numel(vol)==2;sn='06';end
            pw=0;ph=0; if strcmp(vol,'B+');ph=700;end

            % load/crop result
            %DD='/n/coxfs01/donglai/ppl/xupeng/pred-syn/';
            %DD=[D0 'gt-syn/']; % for training
            %syn_warp = h5read(['gt-syn/syn_' vol '_v2_200.h5'],'/main');
            syn_warp = h5read(['results/im_' vol '_v2_200_pred.h5'],'/main');
            %result = h5read(['results/sample_' vol '.hdf'],'/main');
            sz_r = size(syn_warp);
            sz_bd = round((sz_r-[sz{nid}-400 125])/2);
            syn_warp = syn_warp(sz_bd(1)+1:end-sz_bd(1), sz_bd(2)+1:end-sz_bd(2), sz_bd(3)+1:end-sz_bd(3)); 
            % result size = image_v2_200 remove the 200 margin

            pp=cumsum(load(['align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            ww = ceil([max(pp) -min(pp)]); % no need for 200 margin
            % 1250+200*2
            result_o = zeros([1250,1250,125],'uint16');

            for i=1:125
                pd = round(pp(i+14,:)); 
                % reverse case 1:
                tmp = zeros(3072,3072);
                tmp2 = padarray(tmp,[ph,pw],'symmetric','both');
                tmp2((912+pd(1)-ww(1)+ph):(911+pd(1)+1250+ww(3)+ph),...
                        (912+pd(2)-ww(2)+pw):(911+pd(2)+1250+ww(4)+pw)) = syn_warp(:,:,i); 
                result_o(:,:,i) = tmp2(ph+912:ph+911+1250,pw+912:pw+911+1250);
            end
            h5create(['reverse/results_new_' sprintf('%s_%s.h5',vol,suf)],'/main',size(result_o));
            h5write(['reverse/results_new_' sprintf('%s_%s.h5',vol,suf)],'/main',result_o);
            %U_h5write(['../data/cremi/align_v2/syn_' sprintf('%s_%s.h5',vol,suf)],'/main',result_o,5,'uint16');
        end
    case 2.2 % debug
        for nid=4:6
            vol = nn{nid}
            %sn='05';if numel(vol)==2;sn='06';end
            pw=0;ph=0; if strcmp(vol,'B+');ph=700;end

            % load original
            syn_warp = h5read(['results/im_' vol '_v2_200_pred.hdf'],'main');

            % load translated
            sz_r = size(syn_warp);
            sz_bd = round((sz_r-[sz{nid}-400 125])/2);
            syn_warp = syn_warp(sz_bd(1)+1:end-sz_bd(1), sz_bd(2)+1:end-sz_bd(2), sz_bd(3)+1:end-sz_bd(3)); 
            % syn_warp size = image_v2_200 remove the 200 margin
           
            pp=cumsum(load([D0 'align/trans_' vol '_v2.txt']),1);
            pp=-bsxfun(@minus,pp,pp(77,:));
            % 1250+200*2
            ww = ceil([max(pp) -min(pp)])+200;suf='v2_200';
            syn_o = zeros([1250+sum(w([1,3])),1250+sum(w([2,4])),153],'uint16')
            for i=setdiff(1:125,bb{nid}-14+1)
                pd = round(pp(i+14,:)); 

                % check syn_o same with transformed 
                tmp = zeros(3072,3072);
                tmp(912:911+1250,912:911+1250) = syn(:,:,i);
                r_warp = tmp((912+pd(1)-ww(1)):(911+pd(1)+1250+ww(3)),...
                            (912+pd(2)-ww(2)):(911+pd(2)+1250+ww(4))); 
                s1=max(max(abs(r_warp(201:end-200,201:end-200) - double(syn_warp(:,:,i)))));

                % check transformed back with syn
                tmp2 = zeros(3072,3072);
                tmp2 = padarray(tmp2,[ph,pw],'symmetric','both');
                tmp2((912+pd(1)-ww(1)+ph+200):(911+pd(1)+1250+ww(3)+ph-200),...
                        (912+pd(2)-ww(2)+pw+200):(911+pd(2)+1250+ww(4)+pw)-200) = syn_warp(:,:,i); 
                syn_o(:,:,i+14) = tmp2(ph+912:ph+911+1250,pw+912:pw+911+1250);
               
               
            end

        end

    end
end
