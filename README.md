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

