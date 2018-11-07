function [ pred, acc ] = svm_rbf(train_label, test_label,train_data, test_data)
bestmodel = 0;
model = [];
% train svm with rbf kernel
for log2c = -5:2:15
  for log2g = -15:2:13
    cmd = ['-h 0 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    model = svmtrain(train_label,train_data, cmd);
    if (model >= bestmodel)
      bestmodel = model; 
      bestc = 2^log2c; 
      bestg = 2^log2g;
      bestld2g = log2g;
      bestld2c = log2c;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, model, bestc, bestg, bestmodel);
  end
end
cmd = ['-h 0 -b 1 -c ', num2str(bestc), ' -g ', num2str(bestg) ];
model = svmtrain(train_label, train_data, cmd);
[pred, acc, dec_values] = svmpredict(test_label, test_data, model);

end

