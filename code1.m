im = imread('cameraman.tif');
figure;
imshow(im);

% s = size(im);
stepSize = 8;
windowSize = [32 64];

cell = [8 8];
A = imread('cameraman.tif','PixelRegion',{[1,20],[1,22]});
% figure; imshow(A);
[feature, Visualization] = extractHOGFeatures(A);
FeatureSize = length(feature);

harddir = fullfile('harddata');
hardSet = imageSet(harddir,'recursive');
  for y = 1:2:20
      for x = 1:2:22

        RGB = insertShape(im, 'rectangle', [x*10 y*10 32 64]);
        imshow(RGB);
  
      end
  end

% % [testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(hardSet, FeatureSize, cell);
% % Group = svmclassify(SVMStruct,testFeatures);

        

