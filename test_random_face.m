function test_random_face()

clc; close all; 

attr_list = {'Male','Bald','Bangs','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows',...
    'Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','Mustache','Narrow_Eyes',...
    'No_Beard','Pale_Skin','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair', 'Wavy_Hair', ...
    'Wearing_Hat','Wearing_Lipstick','Wearing_Necktie','Young'};

% Add the MatConvnet Path to be able to load and execute the VGG CNN model.
% This should point to the matlab/ folder which contains the scripts which
% we will be using here
addpath('../Deep LFSR/CNN_training/matconvnet-1.0-beta23/matlab/');

% Specify the folder where the VGG model is stored
vgg_model_filename = 'VGG_face_model/vgg-face.mat';

% This specifies the layer from which the features will be extracted
out_layer = 33;

% Define t8he image index to be processed
img_idx = 9;

%--------------------------------------------------------------------------
% Load CNN for training
%--------------------------------------------------------------------------
% Setup mat-conv net
vl_setupnn();

if ~exist(vgg_model_filename,'file')
    error('Make sure to download the vgg-face.mat model\n');
end

%--- Load the model and upgrade it to MatConvNet current version.
fprintf('Loading the CNN model ... \n');
model = load(vgg_model_filename) ; net = model.net;
net = vl_simplenn_tidy(net) ;

% % --- Remove the last two layers
% % Remove the last two layers of the network
% net.layers(end) = []; net.layers(end) = [];
% 
% %--- Insert fully connected layer
% % Insert a fully connected layer
% net.layers{end+1} = struct(...
%     'name','fc8', ...
%     'type','conv',...
%     'weights', {xavier(1,1,4096,40)}, ...
%     'pad', 0, ...
%     'stride',1,...
%     'learningRate',[1,2]);
% %--- Insert softmax layer
% net.layers{end+1} = struct('type','softmax','name','softmax');
% net = vl_simplenn_tidy(net) ;

% Determine the image filename
fprintf('Loading the test image ...\n');
img_filename = sprintf('test_imgs/%d.jpg',img_idx);

% Read the input filename
I = imread(img_filename);

% Show the image
figure(1); imshow(I);

% Derive the crop-coordinates
crop_coord_filename = sprintf('test_imgs/%d.mat',img_idx);

fprintf('Crop the face region ...\n');
if ~exist(crop_coord_filename,'file')
    % Get the corner points to be able to crop the face region
    [x,y] = ginput(2);
    
    save(crop_coord_filename,'x','y');
else
    load(crop_coord_filename);
end

x = round(x); y = round(y);

% Extract the cropped image
Iface = I(y(1):y(2), x(1):x(2),:);
figure(2); imshow(Iface);

% Convert the image to single
im_ = single(Iface) ; % note: 255 range

% Normalize the image by centering
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1);
im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2);
im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3);
    
% Pass the data through the network
res = vl_simplenn(net, im_) ;
    
% Get the input vector (output from the drop out block)
feature = reshape(res(out_layer).x,[4096,1])';

% Determine the number of attributes
Nattr = size(attr_list,2);

for n = 1:Nattr
    % Determine the filename where the svm model is stored
    svm_model_path = sprintf('DATA/SVM_models/%s.mat',attr_list{n});
    % Load the SVM model
    load(svm_model_path);
    % Predict the label
    label = predict(Mdl,feature);
    fprintf('%s: %d\n',attr_list{n},label);
end

