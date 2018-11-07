function [pred, acc ] =gp_ep(train_label, test_label,train_data, test_data )
%SVM_RBF_ONE_VS_ALL Summary of this function goes here
numLabels = max(train_label);
number_test_data = length(test_data(:,1));


loghyper = cell(numLabels,1);

for k=1:numLabels
    loghyper{k} = [0;0];
    labels = double(train_label==k);
    labels(labels ~=1) = -1;
    loghyper{k} = minimize(loghyper{k}, 'binaryEPGP', -40, 'covSEiso', train_data, labels);
    %loghyper{k} = minimize(loghyper, 'binaryLaplaceGP', -20, 'covNNiso', 'cumGauss', train_data, labels);
end

%get probability estimates of test instances using each model
prob = zeros(number_test_data,numLabels);
for k=1:numLabels
    labels = double(train_label==k);
    labels(labels ~=1) = -1;
    [p, mu, s2, nlZ] = binaryEPGP(loghyper{k}, 'covSEiso', train_data, labels, test_data);
    prob(:,k) = p;
end
   
[~,pred] = max(prob,[],2);
acc = sum(pred == test_label) ./ numel(test_label);

end

% newloghyper = minimize(loghyper, 'binaryEPGP', -20, 'covNN', x, y);
%   p4 = binaryEPGP(newloghyper, 'covNN', x, y, t);