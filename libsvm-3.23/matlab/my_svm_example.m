%% https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q10:_MATLAB_interface

% We give the following detailed example by splitting heart_scale into
% 150 training and 120 testing data.  Constructing a linear kernel
% matrix and then using the precomputed kernel gives exactly the same
% testing error as using the LIBSVM built-in linear kernel.

[heart_scale_label, heart_scale_inst] = libsvmread('../heart_scale');

 % Split Data
train_data = heart_scale_inst(1:150,:);
train_label = heart_scale_label(1:150);
test_data = heart_scale_inst(151:270,:);
test_label = heart_scale_label(151:270);
% Linear Kernel
model_linear = svmtrain(train_label, train_data, '-t 0');
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);
model_precomputed = svmtrain(train_label, [(1:150)', train_data*train_data'], '-t 4');
[predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:120)', test_data*train_data'], model_precomputed);
accuracy_L % Display the accuracy using linear kernel
accuracy_P % Display the accuracy using precomputed kernel