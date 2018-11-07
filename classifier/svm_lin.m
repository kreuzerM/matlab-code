function [ pred, acc ] = svm_lin(train_label, test_label,train_data, test_data)
bestmodel = 0;
model = [];
% train svm linear kernel
for log2c = -5:2:15
  
    cmd = ['-t 0 -h 0 -v 5 -c ', num2str(2^log2c)];
    model = svmtrain(train_label, train_data, cmd);
    if (model >= bestmodel)
      bestmodel= model; 
      bestc = 2^log2c; 
      bestld2c = log2c;
    end
    fprintf('%g  %g (best c=%g,  rate=%g)\n', log2c, model, bestc, bestmodel);
 
end
cmd_lin = ['-h 0 -t 0 -b 1 -c ', num2str(bestc), ];
model_lin = svmtrain(train_label, train_data, cmd_lin);
[pred, acc, dec_values] = svmpredict(test_label, test_data, model_lin);



end

