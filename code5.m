%%%% run the classifier on test data  %%%%%%%%%%
%%%%%%%% non-maximum suppression and scaling%%%%

T = imread('image_0188.jpg');

windowW = 128; %%% sliding window 
windowH = 128;

% harddir = fullfile('harddata');
% hardSet = imageSet(harddir,'recursive');
testt = [];
bbox = []; %%% bounding box after non-maximum suppression
for scale = 1.5
   
    T1 = imresize(T,scale);
    T2 = imgaussfilt(T1);
    imageW = size(T2, 2);
    imageH = size(T2, 1);
    U=T2; %%%%%copy of image
    %%% run fixed size window on the given scale of image %%%
    for y = 1:8 :(imageH - windowH + 1)
        for x = 1:8:(imageW - windowW + 1)
            %         window = image(j:j + windowHeight - 1, i:i + windowWidth - 1, :);
            % do stuff with subimage
            RGB1 = insertShape(T2, 'rectangle', [x y windowW windowH]);
            imshow(RGB1);
            ROI1 = T2(y:(y+windowH-1),x:(x+windowW-1)); % subimage
            %          imshow(ROI);
            % Apply pre-processing steps
               lvl1 = graythresh(ROI1);
               ROI1 = im2bw(ROI1, lvl1);
            %%% extract hog features from every window(subimage)
            [hog, visualize] = extractHOGFeatures(ROI1);
            
            %%%% run the classifier on every window of hard data to find false positives%%%
            hardG = svmclassify(svmStruct,hog);
            
            testt = [testt ; hardG]; %#ok<AGROW>
            %          f_array = [f_array ; f]; %#ok<AGROW>
            if(hardG == '1') %%% positive
                T2 = insertShape(T2,'rectangle',[x y windowW windowH], 'Color','green');
                imshow(T2);
                
                [outclass, f] = svmdecision(hog, svmStruct); %%% f gives score
                box = [x y x+windowW y+windowH f]; %input for non-maximum suppression
                bbox = [bbox; box];%#ok<AGROW>
            end
        end
    end
end
%%%%%%%% apply non-maximum suppression %%%%%%%%%%%%%%%%%%%%%%
topp = nms(bbox, 0.3) ;
rows = size(topp,1);

for r = 1:rows
         U = insertShape(U,'rectangle',[topp(r,1) topp(r,2) topp(r,3)-topp(r,1) topp(r,4)-topp(r,2)], 'Color','blue');
        imshow(U);
end





             