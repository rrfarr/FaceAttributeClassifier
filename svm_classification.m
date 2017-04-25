function svm_classification()

clc; close all; clear all;

rng(10); % For reproducibility

% Specify the attribute index
attr_idx = 39; % 9 - black hair

% Specify the path where the data is stored
data_filename = 'DATA/data.mat';

% Load the data
load(data_filename);

% Eactract the feature vectors
vecs = data.feature;

% Use min-max normalization
vecs = minmax(vecs);

% Extract the classification
class = data.attr(:,attr_idx);

%Lambda = logspace(-6,-0.5,11);

% Compute 5-fold classification
CVMdl  = fitclinear(vecs,class,'Solver','sparsa','ObservationsIn','rows',...
    'Regularization','lasso','KFold',5);

% Compute the cross-validation classification error
ce = min(kfoldLoss(CVMdl))

function y = minmax(x)

min_x = repmat(min(x),[size(x,1),1]);
max_x = repmat(max(x),[size(x,1),1]);
y = (x - min_x)./(max_x - min_x);
y = 2*y -1;