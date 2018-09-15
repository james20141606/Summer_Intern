% check alignment method

D0='../data/cremi/';
Da='db/align/';
    
nn={'A','B','C','A+','B+','C+'};
tid=3.6
switch floor(tid)
case 3 % hack A+ crack
    ind = 14+[1,34];
    switch tid
    case 3 % generate frame
        vol = nn{4}
        suf = 'v2_200';
        out = h5read(['images/im_' sprintf('%s_%s.h5',vol,suf)],'/main');
        for i=ind
            for j=-1:1
                imwrite(out(:,:,i+j),sprintf('crack/im%03d.png',i+j));
            end
        end
    case 3.1 % get separate crack region 
        for i = ind
            im = imread(sprintf('crack/im%03d.png',i));
            [seg,smax] = bwlabel(im>70);
            sc = histc(seg(:),1:smax);
            [~,sid] = sort(sc,'descend');
            for j=1:2
                rr = imfill(seg==sid(j),'holes');
                imwrite(im.*uint8(rr),sprintf('crack/im%03d_p%d.png',i,j));
            end
        end
    case 3.2 % do affine align
        % python T_zudi_0516.py 3 
    case 3.3 % apply warp
        vid=0;
        for i = ind
            for j = [1,2]
                im = imread(sprintf('crack/im%03d_p%d.png',i,j));
                sz = size(im);
                sprintf('crack/align_%d_%d_%d.hdf',vid,i,j)
                tmp = h5read(sprintf('crack/align_%d_%d_%d.hdf',vid,i,j),'/main');
                if j==1
                    B = imwarp(im, affine2d(tmp(:,:,2)),'FillValues',0,'OutputView',imref2d(sz));
                    disp(tmp(:,:,2))
                    disp(affine2d(tmp(:,:,2)))
                else
                    B = B+imwarp(im, affine2d(tmp(:,:,2)),'FillValues',0,'OutputView',imref2d(sz));
                end
            end
            imwrite(B,sprintf('crack/im%03d_warp%d.png',i,vid))
        end
    case 3.4 % visualization
        suf='';sn='_orig';
        suf='_warp0';sn='_warp';
        suf='_warp0_ip_db';sn='_ip';
        for i = ind
            out=cell(1,3);
            for j=[1 3]
                out{j} = imresize(imread(sprintf('crack/im%03d.png',i+j-2)),0.3);
            end
            out{2} = imresize(imread(sprintf('crack/im%03d%s.png',i,suf)),0.3);
            U_gifWrite(cat(4,out{:}),['crack' sn '_' num2str(i) '.gif']);
        end
    case 3.5 % generate frame
        vol = nn{4}
        suf = 'v2_200';
        out = h5read(['align_v2/im_' sprintf('%s_%s.h5',vol,suf)],'/main');
        for i=ind
            out(:,:,i) = imread(sprintf('crack/im%03d_warp0_ip_db.png',i));
        end
        h5create(['crack/im_' sprintf('%s_%s_nocrack.h5',vol,suf)],'/main',size(out));
        h5write(['crack/im_' sprintf('%s_%s_nocrack.h5',vol,suf)],'/main',out);
    case 3.6 % reverse bigger crack left or right to origin
        vid=0;
        disp(ind(1))
        for i = ind
            for j = [1,2]
                im = imread(sprintf('crack/im%03d_warped%d.png',i,j));
                sz = size(im);
                sprintf('crack/align_%d_%d_%d.hdf',vid,i,j)
                tmp = h5read(sprintf('crack/align_%d_%d_%d.hdf',vid,i,j),'/main');
                tmpp = tmp(:,:,2)
                if j==1
                    disp(tmp(:,:,2))
                    disp(tmpp)
                    
                    B = imwarp(im, invert(affine2d(tmpp)),'FillValues',0,'OutputView',imref2d(sz));
                else
                    B = B+imwarp(im, invert(affine2d(tmpp)),'FillValues',0,'OutputView',imref2d(sz));
                end
            end
            imwrite(B,sprintf('crack/im%03d_reverse%d.png',i,vid))
        end
    end
end
