%%%%%%%%% code for extracting hog features from training data and training
%%%%%%%%% SVM Classifier--------- initial training%%%%%%%%%%%%%%%%%%%%%%%%

 img = imread('car_0001.ppm');
 [featureVector, hogVisualization] = extractHOGFeatures(img);
 hogFeatureSize = length(featureVector);
 cellSize = [8 8];
 
carsdir = fullfile('dataset'); % read training dataset
% testdir = fullfile('testdata');% read test dataset

trainingSet = imageSet(carsdir,'recursive');
% testSet = imageSet(testdir,'recursive');

trainingFeatures = []; %for svm training
trainingLabels   = [];
for digit = 1:numel(trainingSet) % two training sets 1(positive) and 0(negative)

    numImages = trainingSet(digit).Count; % number of images in each training set
    features  = zeros(numImages, hogFeatureSize, 'single');
    for i = 1:numImages
        
        img = read(trainingSet(digit), i);
            img = imresize(img, [128 128]);
        % Apply pre-processing steps
        lvl = graythresh(img);
        img = im2bw(img, lvl);
    
        % extract hog features of each image in training set
        features(i, :) = extractHOGFeatures(img);
    end
     labels = repmat(trainingSet(digit).Description, numImages, 1);
     trainingFeatures = [trainingFeatures; features];   %#ok<AGROW>
     trainingLabels   = [trainingLabels;   labels  ];   %#ok<AGROW>
end

        % train linear-SVM Classifier on training data
     SVMStruct =  svmtrain(trainingFeatures, trainingLabels);

     % run the classifier on test data  
% [testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);
% Group = svmclassify(SVMStruct,testFeatures);

    