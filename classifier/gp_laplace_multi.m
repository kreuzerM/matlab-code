function [pred, acc ] =gp_laplace( train_label, test_label,train_data, test_data )

numLabels = max(train_label);
number_test_data = length(test_data(:,1));

para.kernel = @covSEiso;
para.hyp = log([ones(1,1)*2, 10]); % initilization of hyper-parameters
para.S = number_test_data; % sample number

para.c = numLabels; % numble of categories
para.Ncore = 12; % multiple CPU cores parallelizing
para.flag = false; % plotting flag
hyp = para.hyp;

% hyper-parameter optimization
[ hyp ] = modelSelection(para, train_data, train_label);

% compute multi-class GP kernel
[ K ] = covMultiClass(hyp, para, train_data, []);

% estimate the posterior probility of p(f|X,Y)
[ gp_model ] = LaplaceApproximation(hyp, para, K, Xtrain, ytrain);

% save GP parameters
save('classifier_gp_demo.mat','gp_model','gp_para');

% prediction p(y*|X,y,x*)
[ pred prob fm ] = predictGPC(hyp, para, train_data, train_label, gp_model, test_data);
acc = sum(pred == test_label) ./ numel(test_label);
end

