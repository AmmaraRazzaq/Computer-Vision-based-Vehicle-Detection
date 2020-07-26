% II = imread('cars.jpg');
% 
% %%%%%%% image scales%%%%%%%%
% for scale=0.5:0.1:1.0
%     JJ = imresize(II,scale);
%     KK = imgaussfilt(JJ);
%     figure;
%     imshow(KK);
% end
II = imread('image0001.jpg');
% imshow(II);
  % Apply pre-processing steps
%         lvlt = graythresh(II);
%         II = im2bw(II, lvlt);
II = integralImage(II);
 [hogI,visI]=extractHOGFeatures(II);
 plot(visI);
%  img = imresize(II, [128 128]);
%  imshow(II);
%   threshold = graythresh(II);
%   img = im2bw(II, threshold);
% imshow(img);
% [hog,vis]=extractHOGFeatures(II);
% plot(vis);

% 
%     I = read(hardSet,1);
%     imshow(I);

svmStruct =  svmtrain(trainingFeatures, trainingLabels);
