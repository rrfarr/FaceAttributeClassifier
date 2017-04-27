function svm_learn_models()
% This function is used to learn the SVM model for all the attributes

clc; close all; clear all;

attr_str = {'5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes', ...
    'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair', 'Blurry', ...
    'Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee', ...
    'Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open',...
    'Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose', ...
    'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair', ...
    'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace',...
    'Wearing_Necktie','Young'};


rng(10); % For reproducibility

% Specify the path where the data is stored
data_filename = 'DATA/data.mat';

% Load the data
load(data_filename);

% Eactract the feature vectors
vecs = data.feature;

% Use min-max normalization
vecs = minmax(vecs);

fid = fopen('RESULTS/attribute_classification.txt','w');
for attr_idx = 1:40
    % Determine the filename where the SVM model will be stored
    model_filename = sprintf('DATA/SVM_models/%s.mat',attr_str{attr_idx});
    
    % Extract the attribute class
    class = data.attr(:,attr_idx);
    
    fprintf('Cross-validation for feature %s ...\n',attr_str{attr_idx});
    % Compute 5-fold classification
    CVMdl  = fitclinear(vecs,class,'Solver','sparsa','ObservationsIn','rows',...
    'Regularization','lasso','KFold',5);
    % Compute the cross-validation c;lassification error
    ce = min(kfoldLoss(CVMdl));
    
    fprintf(1,'%s CE: %0.4f\n',attr_str{attr_idx},ce);
    fprintf(fid,'%s CE: %0.4f\n',attr_str{attr_idx},ce);
    
    fprintf('Learning the linear SVM model on all data ...\n');
    % Fit a linear model using all the data - this will be our model
    Mdl = fitclinear(vecs,class,'Solver','sparsa','ObservationsIn','rows',...
    'Regularization','lasso');

    fprintf('Save the SVM model ...\n');
    save(model_filename,'Mdl');
    fprintf('..........................................\n');
end

fclose('all');


function y = minmax(x)

min_x = repmat(min(x),[size(x,1),1]);
max_x = repmat(max(x),[size(x,1),1]);
y = (x - min_x)./(max_x - min_x);
y = 2*y -1;