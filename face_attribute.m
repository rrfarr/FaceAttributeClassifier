function face_attribute()
% This function will be used to exploit the possibility of using the VGG
% face model for attribute face recognition

clc; close all; clear all;

% Add the MatConvnet Path to be able to load and execute the VGG CNN model.
% This should point to the matlab/ folder which contains the scripts which
% we will be using here
addpath('../Deep LFSR/CNN_training/matconvnet-1.0-beta23/matlab/');

% Specify the location where the metadata will be stored. This file will
% contain information extracted from the annotation files provided with the
% celebA dataset. This file is initially not available but will be created
% in the first run.
metadata_filename = 'DATA/metadata.mat';

% Specify the folder where the VGG model is stored
vgg_model_filename = 'VGG_face_model/vgg-face.mat';

% This specifies the layer from which the features will be extracted
out_layer = 33;

%--------------------------------------------------------------------------
% Load CNN for training
%--------------------------------------------------------------------------
% Setup mat-conv net
vl_setupnn();

if ~exist(vgg_model_filename,'file')
    error('Make sure to download the vgg-face.mat model\n');
end

%--- Load the model and upgrade it to MatConvNet current version.
model = load(vgg_model_filename) ; net = model.net;
net = vl_simplenn_tidy(net) ;

% --- Remove the last two layers
% Remove the last two layers of the network
net.layers(end) = []; net.layers(end) = [];

%--- Insert fully connected layer
% Insert a fully connected layer
net.layers{end+1} = struct(...
    'name','fc8', ...
    'type','conv',...
    'weights', {xavier(1,1,4096,40)}, ...
    'pad', 0, ...
    'stride',1,...
    'learningRate',[1,2]);
%--- Insert softmax layer
net.layers{end+1} = struct('type','softmax','name','softmax');
net = vl_simplenn_tidy(net) ;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Load the data
%--------------------------------------------------------------------------
if ~exist(metadata_filename,'file')
    % Derive the list of train/val images to be considered
    fprintf('Parse the evaluation file ...\n');
    [img_list, eval] = parse_eval_file('CelebA/Eval/list_eval_partition.txt');

    % Put this information as part of the metadata
    metadata.img_list = img_list; 
    metadata.eval     = eval;
    % Get the attributes
    fprintf('Load the attributes ... \n');
    metadata = get_attributes('CelebA/Anno/list_attr_celeba.txt',metadata);

    % Save the metadata
    save(metadata_filename,'metadata');
else
    load(metadata_filename);
end

% Extract information from metadata file
img_list = metadata.img_list;
eval     = metadata.eval;
attr     = metadata.attr;

% Determine the number of samples
N = size(eval,2);

% Initialize the data
for n = 1:N
    % Determine the filename
    filename = sprintf('DATASET/CelebA/Img/img_align_celeba/%s',img_list{n});
    
    if ~exist(filename,'file')
        continue;
    end
    
    % Load the image
    im = imread(filename);
    
    % Load a test image from Wikipedia and run the model.
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1);
    im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2);
    im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3);
    
    % Pass the data in the network
    res = vl_simplenn(net, im_) ;
    
    % Get the input vector (output from the drop out block)
    feature(n,:) = reshape(res(out_layer).x,[4096,1])';
    eval2(n,1) = eval(n);
    attr2(n,:) = attr{n};
    img_list2{n} = img_list{n};
    
     fprintf('%d out of %d ready\n',n,N);
end

data.eval = eval2;
data.img_list = img_list2;
data.attr = attr2;
data.feature = feature;

% Save the data including features
save('DATA/data.mat','data');

% 
% for idx = 1:100
%     im = imread(sprintf('DATASET/CelebA/Img/img_celeba/%0.6d.jpg',idx));
% 
%     % Load a test image from Wikipedia and run the model.
%     im_ = single(im) ; % note: 255 range
%     im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
%     im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1);
%     im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2);
%     im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3);
% 
%     res = vl_simplenn(net, im_) ;
%     
%     figure(1);
%     subplot(1,2,1);
%     imshow(imresize(im,[224,224]));
%     subplot(1,2,2);
%     imshow(mat2gray(imresize(mean(res(29).x,3),[224,224])));
%     waitforbuttonpress
%     
% end

% Get the attributes
function metadata = get_attributes(filename,metadata)
fid = fopen(filename,'r');

k = 1; mask = zeros(size(metadata.eval));

img_list = metadata.img_list;

while ~feof(fid)
    % Read the line
    line = fgetl(fid);
    
    if k > 2
        % Remove double spacing
        line = regexprep(line,' +',' ');

        % Find the index of spaces
        idx = strfind(line,' ');
        
        % Parse the filename to which the attributes belong
        filename = line(1:idx(1)-1);
        
        idx_mask = find(mask == 0);
        
        for i = 1:size(idx_mask,2)
            temp_filename = img_list(idx_mask(i));
            
            if strcmp(temp_filename,filename)
                for j = 2:size(idx,2)
                    metadata.attr{idx_mask(i)}(1,j-1) = str2double(line(idx(j-1)+1:idx(j)-1));
                end
                metadata.attr{idx_mask(i)}(1,size(idx,2)) = str2double(line(idx(end)+1:end));
                % Set the mask to 1 so that we do not search next time
                mask(idx_mask(i)) = 1;
                break;
            end
        end
        
    end
    
    if ~mod(k-2,1000)
        fprintf('%d attributes loaded\n',k-2);
    end
    k = k + 1;
end

function [img_list, eval] = parse_eval_file(filename)
fid = fopen(filename,'r');

k = 1;
while ~feof(fid)
    line = fgetl(fid);
    
    idx = strfind(line,'jpg') + 3;
    
    flag = str2double(line(idx:end));
    
    if flag == 0 
        img_list{k} = line(1:idx-1);
        eval(k) = 1;
        k = k + 1;
    elseif flag == 2
        img_list{k} = line(1:idx-1);
        eval(k) = 2;
        k = k + 1;
    end
end

function weights = xavier(varargin)
%XAVIER  Xavier filter initialization.
%   WEIGHTS = XAVIER(H, W, C, N) initializes N filters of support H x
%   W and C channels using Xavier method. WEIGHTS = {FILTERS,BIASES}is
%   a cell array containing both filters and biases.
%
% See also:
% Glorot, Xavier, and Yoshua Bengio.
% "Understanding the difficulty of training deep feedforward neural networks."
% International conference on artificial intelligence and statistics. 2010.

filterSize = [varargin{:}] ;
scale = sqrt(2/prod(filterSize(1:3))) ;
filters = randn(filterSize, 'single') * scale ;
biases = zeros(filterSize(4),1,'single') ;
weights = {filters, biases} ;

