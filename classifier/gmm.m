function [pred, acc ] =gmm(train_label, test_label,train_data, test_data )
%gmm( train_label, test_label,svm_training_data, svm_test_data );
%SVM_RBF_ONE_VS_ALL Summary of this function goes here
numLabels = max(train_label);
number_test_data = length(test_data(:,1));
% train GMMS 
 maxiter = 1000;
 mu =[];
 sigma = [];
 for k = 1:numLabels
    idx = find(train_label==k);
    tmp = train_data(idx,:);
    [mu(k,:),tsigma] = EM(tmp,1,maxiter);
    sigma{k}= cell2mat(tsigma);
 end
% test GMMS

pred= [];

prob = zeros(number_test_data,numLabels);
for k = 1:numLabels
    
     prob(:,k)= gaussianND(test_data, mu(k, :), sigma{k});
   
end
[~,pred] = max(prob,[],2);
acc = sum(pred == test_label) ./ numel(test_label);  
end

% maxiter = 1000;
%  mu =[];
%  sigma = [];
%  training_data_comp = [];
%  test_data_comp = [];
%  for q=1:cfg.n_src
%     training_data_comp = [ training_data_comp ;squeeze(mod_data(:,:,q))];
%     test_data_comp = [ test_data_comp;squeeze(mod_test_data(:,:,q))];
% end
%  for q = 1:cfg.n_src
%     tmp = mod_data(:,:,q);
%     [mu(q,:),tsigma] = EM(tmp,1,maxiter);
%     sigma{q}= cell2mat(tsigma);
%  end
% % test GMMS
% confusion_matrix = zeros(cfg.n_src,cfg.n_src);
% gmm_pred_label = [];
% 
% for q = 1:cfg.n_src
%     tdata = squeeze(mod_test_data(:,:,q));
%     probability_matrix = zeros(cfg.n_src,length(test_data_comp(:,1)));
%     %probability_matrix = zeros(cfg.n_src,length(svm_test_data(:,1)));
%     for n = 1:cfg.n_src  
%         probability_matrix(n,:)= gaussianND(test_data_comp, mu(n, :), sigma{n});
%     end
%    [~,ix] = max(probability_matrix,[],1);
%    for i=1:length(ix)
%         confusion_matrix(ix(i),q) = confusion_matrix(ix(i),q) +1 ;
%    end
% end