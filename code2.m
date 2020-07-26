%%%%%%%%%% sliding window technique and hard mining %%%%

% I = imread('camera.png');
% J=I;
% imageWidth = size(I, 2);
% imageHeight = size(I, 1);

windowWidth = 128; %%% sliding window
windowHeight = 128;

harddir = fullfile('harddata');
hardSet = imageSet(harddir);
nImages = hardSet.Count;

for h = 1:nImages
    I = read(hardSet,h);
    I = imresize(I,[400 270]);
    J=I;
    imageWidth = size(I, 2);
    imageHeight = size(I, 1);
    
    %%% run fixed size window on the given image %%%
    for j = 1:8 :(imageHeight - windowHeight + 1)
        for i = 1:8:(imageWidth - windowWidth + 1)
            %         window = image(j:j + windowHeight - 1, i:i + windowWidth - 1, :);
            % do stuff with subimage
            
	    RGB = insertShape(I, 'rectangle', [i j windowWidth windowHeight]);
            imshow(RGB);
            ROI = I(j:(j+windowHeight-1),i:(i+windowWidth-1)); % subimage = region covered by sliding window
            % Apply pre-processing steps
            lvl2 = graythresh(ROI);
            ROI = im2bw(ROI, lvl2);
            %%% extract hog features from every window(subimage)
            [hog, visualize] = extractHOGFeatures(ROI);
            
            %%%% run the classifier on every window of hard data to find false positives%%%
            hardGroup = svmclassify(SVMStruct,hog);
            
            if(hardGroup == '1') %%% false-positive detection
               
                I = insertShape(I,'rectangle',[i j windowWidth windowHeight], 'Color','green');
                imshow(I);
                trainingFeatures = [trainingFeatures ; hog]; %#ok<AGROW>
                trainingLabels = [trainingLabels ; '0']; %#ok<AGROW>
            end
        end
    end
end

%%%% find windows which give false-positive and retrain classifier --- HARD MINING %%%
%%%% classifier is retrained using positive, negative and hard dataset %%%
svmStruct =  svmtrain(trainingFeatures, trainingLabels);

