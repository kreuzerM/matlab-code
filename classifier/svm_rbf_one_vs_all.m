function [pred, acc ] = svm_rbf_one_vs_all( train_label, test_label,train_data, test_data )
%SVM_RBF_ONE_VS_ALL Summary of this function goes here
numLabels = max(train_label);
number_test_data = length(test_data(:,1));
best_model = [];
bestc = 0;
bestg = 0;
cmd=0;
best_model = cell(numLabels,1);
bestmodel = 0;
for k=1:numLabels
    labels = double(train_label==k);
    labels(labels ~=1) = -1;
    if(k==1)
        for log2c = -5:2:15
            for log2g = -15:2:13
            cmd = ['-h 0 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            model = svmtrain(labels, train_data, cmd);
                if (model >= bestmodel)
                    bestmodel = model; 
                    bestc = 2^log2c; 
                    bestg= 2^log2g;
                end
            end
        end
    end
    % to do: optimize Parameters for each model
    %cmd = [' -b 1 -h 1 -v 5 -c ', num2str(bestc), ' -g ', num2str(bestg)];
    cmd = ['-h 0  -c ', num2str(bestc), ' -g ', num2str(bestg),' -b 1' ];
    
    best_model{k} = svmtrain(labels, train_data, cmd);
end

%get probability estimates of test instances using each model
prob = zeros(number_test_data,numLabels);
for k=1:numLabels
    test_labels = double(test_label==k);
    test_labels(test_labels ~=1) = -1;
    %[predict_label, accuracy, dec_values] = svmpredict(test_label, svm_test_data, model);
    [~,~,p] = svmpredict(test_labels, test_data, best_model{k});
    %prob(:,k) = p(:,best_model{k}.Label==1);    % probability of class==k
    prob(:,k) = p;
end
   
[~,pred] = max(prob,[],2);
 acc = sum(pred == test_label) ./ numel(test_label);

end

