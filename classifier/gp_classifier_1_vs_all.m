function [pred, acc ] =gp_classifier_1_vs_all( train_label, test_label,train_data, test_data )

numLabels = max(train_label);
number_test_data = length(test_data(:,1));
feat_dim = length(test_data(1,:));
test_labels = ones(length(test_label),1);
numClassifier = numLabels*(numLabels-1)./2
hyper = cell(numClassifier,1);
prob = zeros(number_test_data,numClassifier);
voting_matrix = zeros(number_test_data,numLabels);
covfunc = @covSEiso;
meanfunc =  @meanZero;
likfunc = @likErf;
inference = @infLaplace;
%inference = @infGP;
l=1;
for k=1:numLabels
    for j=k+1:numLabels
    
    param.mean = [];
    param.cov = zeros(2,1);
    ixk = find(train_label == k);
    ixj = find(train_label == j);
    train_data_k = train_data(ixk,:);
    train_data_j = train_data(ixj,:);
    train_data_kj = [train_data_k;train_data_j];
    train_labels = [ones(length(ixk),1);-1*ones(length(ixj),1)];
    param =minimize(param, @gp, -40, inference, meanfunc, covfunc, likfunc, train_data_kj, train_labels);
    hyper{l}=param;
   [ymu ys2 fmu fs2 lp,post] = gp(hyper{l}, inference, meanfunc, covfunc, likfunc, train_data_kj, train_labels, test_data, test_labels);
    p = exp(lp);
    p1_idx = find(p >= 0.7);
    p2_idx = find(p <= 0.3);
    if(length(p1_idx) > 0)
        voting_matrix(p1_idx,k) = voting_matrix(p1_idx,k) +1;
    end
    if(length(p2_idx) > 0)
        voting_matrix(p2_idx,j) = voting_matrix(p2_idx,j) +1;
    end
    l = l+1;
    end  
end  
[~,pred] = max(voting_matrix,[],2);
acc = sum(pred == test_label) ./ numel(test_label);

end
% meanfunc = @meanConst; hyp.mean = 0;
% covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
%   likfunc = @likErf;
% 
%   hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
%   [a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));

