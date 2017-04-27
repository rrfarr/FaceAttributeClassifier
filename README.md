# FaceAttributeClassifier
This project implements an attributed classification using features from the VGG face recognizer and linear SVMs.

face_attribute()
----------------

The face_attribute() script is used to extract the feature 
descriptors using the VGG face model. In order to run this
function you must install matconvnet and update the path 
within addpath (line 8) accordingly.

The DATA/metadata.mat contains data structure which contains a list of images that will be used in the experiment, eval which specifies which data is to be considered for training (1) and which for testing (2) and the extracted attributes. One must note that for now we are considering the training data which is a cropped face and no face detection is provided. 

This function will pass each training image from the celebA dataset throgh the VGG CNN and extracts the feature at layer 33. Both the features and attributes of each training data is stored in a structure DATA/data.mat.

svm_learn_models()
------------------

This script is used to load the training features and corresponding attributes and learn an SVM for each attribute (40 in tolal). Each model is stored in DATA/SVM_models. The results outputs the 5-fold cross-validation error which can be used to decide which attributes are being classified correctly and which not and also the model trained on all the data. This model will then be used later on to predict the class labels.

test_random_face()
------------------

This script is basically an attribute prediction demo where a number of test images are contained within the test_imgs/ folder. The img_idx specifies the index of the file to be considered. The user is asked to specify the top-left and bottom right coordinates using the mouse on the first run. It will then store the coordinates so when you run it the other times it will load the cropped coordinates automatically. This demo prints a list of attributes and their prediction according to the loaded SVM model. Please note that +1 indicates positive class and -1 indicates negative class.

