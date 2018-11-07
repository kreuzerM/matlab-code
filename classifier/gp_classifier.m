function [pred, acc ] =gp_classfier( train_label, test_label,train_data, test_data )

numLabels = max(train_label);
number_test_data = length(test_data(:,1));
feat_dim = length(test_data(1,:));
test_labels = ones(length(test_label),1);
prob = zeros(number_test_data,numLabels);

hyper = cell(numLabels,1);
%covfunc = @covSEiso; %@covRQiso 
covfunc = @covSEiso;
%covfunc = @covSEiso;
%covfunc = @covSEard;
meanfunc =  @meanZero;
%likfunc = @likErf;
likfunc = @likLogistic;
%inference = @infLOO;
inference = @infLaplace;
for k=1:numLabels
    param.mean = [];
    param.cov = zeros(2,1);
    
%     param.cov = zeros(feat_dim+1,1);
    train_labels = double(train_label==k);
    train_labels(train_labels ~=1) = -1;
    %[hyper, fX, i] = minimize(hyper, 'binaryGP', length, 'approxEP', 'covSEiso', 'logistic', x, y);
    param =minimize(param, @gp, -50, inference, meanfunc, covfunc, likfunc, train_data, train_labels);
    hyper{k}=param;
    [ymu ys2 fmu fs2 lp,post] = gp(hyper{k}, inference, meanfunc, covfunc, likfunc, train_data, train_labels, test_data, test_labels);
    p = exp(lp);
    prob(:,k) = p;
end   
[~,pred] = max(prob,[],2);
acc = sum(pred == test_label) ./ numel(test_label);

end
% meanfunc = @meanConst; hyp.mean = 0;
% covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
%   likfunc = @likErf;
% 
%   hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
%   [a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
