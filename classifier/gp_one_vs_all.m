function [pred,acc] = GP( train_label, test_label, train_data, test_data )
%GP Summary of this function goes here
%   Detailed explanation goes here
numLabels = max(train_label);
hyper = cell(numLabels,1);
for k=1:numLabels
     labels = train_label==1;
     labels(labels ~=1) = -1;
    loghyper = [0 0];
    hyper{k} = minimize(loghyper, 'binaryLaplaceGP', -20, 'covSEiso', 'cumGauss', train_data, labels)
end

 for k=1:cfg.n_src
     labels 
    p = binaryLaplaceGP(hyper{k}, 'covSEiso', 'cumGauss', train_data, labels, test_data);
    
    end
   
[~,pred] = max(prob,[],2);
 acc = sum(pred == test_label) ./ numel(testLabel)

end

