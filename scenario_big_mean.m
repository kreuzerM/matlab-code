clean;
config;
tstart = tic;
%% setup room, sensors and source positions
cfg.room_dim = [ 6 6 3.5];
cfg.n_array = 4;
cfg.n_mic = 2;
cfg.pos_ref = [ 3 0.5 1.5; 0.5 3.5 1.5; 3 5.5 1.5; 5.5 3.5 1.5];
cfg.mic_array_rot = [ 90,0,-90,-180];
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
% generate mic positions for each array
for i=1:cfg.n_array
    cfg.mic_pos(:,:,i) = generateSensorArray(cfg.pos_ref(i,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(i));
end
%% source positions
cfg.n_src = 6;
cfg.source_pos = [2.9 1.5 1.5; 1.5 2.5 1.5; 4.2 2.5 1.5; 3.35 4 1.5;...
                 2.65 3.7 1.5;4.3 4.2 1.5;]
             %; 4.38 4.4 1.5; 1.5 3.9 1.5; 4.33 6 1.5;
                 %3.2 5.5 1.5;1 5.75 1.5];
%% visualize Room
visualizeSetup(cfg,1);
%% training & test source signals 
cfg.sig_len = 3;
training_source_path_idx = [10,9,15,5,1,8,13];  %,8,11,7,13,3];
test_source_path_idx = [2,4,6,12,14,11]; %2,4,6,12,14];
test_signals = zeros(cfg.fs*cfg.sig_len,10);
training_signals = zeros(cfg.fs*cfg.sig_len,cfg.n_src);

for i=1:cfg.n_src
    training_source_path = cfg.source_path{training_source_path_idx(i)};
    test_source_path = cfg.source_path{test_source_path_idx(i)};
    training_signals(:,i)=getSourceSignal(training_source_path,cfg.fs,cfg.sig_len);
    test_signals(:,i)=getSourceSignal(test_source_path,cfg.fs,cfg.sig_len);
end

%% generate training data
data = [];
number_training_data = 250;
true_pos = cfg.source_pos;
training_snr = zeros(number_training_data,1);
training_t60 = zeros(number_training_data,1);
%training_pos_offset =zeros(number_training_data,2);

train_pos_s1 = zeros(number_training_data,2);
train_pos_s2 = zeros(number_training_data,2);
train_pos_s3 = zeros(number_training_data,2);
train_pos_s4 = zeros(number_training_data,2);
train_pos_s5 = zeros(number_training_data,2);
train_pos_s6 = zeros(number_training_data,2);

for i = 1:number_training_data
    % introduce some variance to source positions
    %offset = 0.4 +(2*randn(cfg.n_src,2)-1);
    offset = 0.5 *(2*rand(cfg.n_src,2)-1);
    cfg.source_pos(:,1:2) = true_pos(:,1:2) + offset;
    % vary the SNR and T60
    SNR = [-5,0,5,10,15,20,25,30];
    T60 = 0.4;
    %T60 = [ 0.25,0.3,0.35,0.4,0.45,0.5];
    T60_idx = randperm(length(T60));
    snr_idx = randperm(length(SNR));
    cfg.SNR = SNR(snr_idx(1));
    cfg.beta = T60(T60_idx(1));
    training_snr(i) = SNR(snr_idx(1));
    training_t60(i) = T60(T60_idx(1));
    train_pos_s1(i,:) =cfg.source_pos(1,1:2);
    train_pos_s2(i,:) = cfg.source_pos(2,1:2);
    train_pos_s3(i,:) = cfg.source_pos(3,1:2);
    train_pos_s4(i,:) = cfg.source_pos(4,1:2);
    train_pos_s5(i,:) = cfg.source_pos(5,1:2);
    train_pos_s6(i,:) = cfg.source_pos(6,1:2);
    
    % generate noisy microphone signals (dim: samples x mics x sources x
    % nodes)
    [xnoisy,H] = generateMicrophoneSignals(training_signals,cfg);
    % generate STFT signals (dim: fbins x tbins x mics x sources x nodes
    Xnoisy = computeSTFT(xnoisy,cfg);
    PRPs = (PRP(Xnoisy,cfg));
    MSCs = MSC(Xnoisy,cfg);
    [Diff,CDR] = CDRDIFF(Xnoisy,cfg);
    DOAs = getDOAs(xnoisy,cfg,1);
    delay = gcc_phat(xnoisy,cfg);
    vec=[DOAs,Diff,CDR,MSCs,real(PRPs),imag(PRPs),delay];
    vec = vec.';
    data(i,:,:) = vec;
end

test_data = [];
number_test_data = 100;
test_snr = zeros(number_test_data,1);
test_t60 = zeros(number_test_data,1);
%test_pos_offset =zeros(number_test_data,2);
test_pos_s1 = zeros(number_test_data,2);
test_pos_s2 = zeros(number_test_data,2);
test_pos_s3 = zeros(number_test_data,2);
test_pos_s4 = zeros(number_test_data,2);
test_pos_s5 = zeros(number_test_data,2);
test_pos_s6 = zeros(number_test_data,2);


for i = 1:number_test_data
    %choose random SNR and T60
    offset = 0.5 *(2*rand(cfg.n_src,2)-1);
    cfg.source_pos(:,1:2) = true_pos(:,1:2) + offset;
    SNR = [-5,0,5,10,15,20,25,30];
    %SNR = [18,25,22];
    T60 = [0.4];
    %T60 = [ 0.25,0.3,0.35,0.4,0.45,0.5];
    T60_idx = randperm(length(T60));
    snr_idx = randperm(length(SNR));
    cfg.SNR = SNR(snr_idx(1));
    cfg.beta = T60(T60_idx(1));
    test_snr(i) = SNR(snr_idx(1));
    test_t60(i) = T60(T60_idx(1));
    %test_pos_offset(i,:) = offset;
    test_pos_s1(i,:) =cfg.source_pos(1,1:2);
    test_pos_s2(i,:) = cfg.source_pos(2,1:2);
    test_pos_s3(i,:) = cfg.source_pos(3,1:2);
    test_pos_s4(i,:) = cfg.source_pos(4,1:2);
    test_pos_s5(i,:) = cfg.source_pos(5,1:2);
    test_pos_s6(i,:) = cfg.source_pos(6,1:2);
    
    [xnoisy,H] = generateMicrophoneSignals(training_signals,cfg);
    Xnoisy = computeSTFT(xnoisy,cfg);
    PRPs = (PRP(Xnoisy,cfg));
    MSCs = MSC(Xnoisy,cfg);
    [Diff,CDR] = CDRDIFF(Xnoisy,cfg);
    DOAs = getDOAs(xnoisy,cfg,1);
    delay = gcc_phat(xnoisy,cfg);
    vec=[DOAs,Diff,CDR,MSCs,real(PRPs),imag(PRPs),delay];
    vec = vec.';
    test_data(i,:,:) = vec;
end
run_time = toc(tstart);
test_snr = repmat(test_snr,number_test_data,1);
test_t60 = repmat(test_t60,number_test_data,1);
%test_pos = repmat(test_pos_offset,number_test_data,1);
training_snr = repmat(training_snr,number_training_data,1);
training_t60 = repmat(training_t60,number_training_data,1);
%training_pos = repmat(tra_pos_offset,number_training_data,1);
%% evaluation
%% select only certain features
feat_vec = [1:8,13:28];
mod_data = data(:,feat_vec,:);
mod_test_data = test_data(:,feat_vec,:);
% train GMMS 
 maxiter = 1000;
 mu =[];
 sigma = [];
 for q = 1:cfg.n_src
    tmp = mod_data(:,:,q);
    [mu(q,:),tsigma] = EM(tmp,1,maxiter);
    sigma{q}= cell2mat(tsigma);
 end
% test GMMS
confusion_matrix = zeros(cfg.n_src,cfg.n_src);
for q = 1:cfg.n_src
    tdata = squeeze(mod_test_data(:,:,q));
    probability_matrix = zeros(cfg.n_src,length(tdata(:,1)));
    for n = 1:cfg.n_src  
        probability_matrix(n,:)= gaussianND(tdata, mu(n, :), sigma{n});
    end
   [~,ix] = max(probability_matrix,[],1);
   for i=1:length(ix)
        confusion_matrix(ix(i),q) = confusion_matrix(ix(i),q) +1 ;
   end
end

% plot confusion matrix
figure('Name','GMM');
plotConfMat(confusion_matrix, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring '});

% plotConfMat(confusion_matrix, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring ','7 keyboard ',...
%     '8 vacuum cleaner ','9 coughing ','10 Male3 '});



%% SVM part
% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
% 
% The k in the -g option means the number of attributes in the input data.


train_label =  ones(length(data),1) * [1:cfg.n_src];
test_label =  ones(length(tdata),1) * [1:cfg.n_src];
train_label = train_label(:);
test_label = test_label(:);
svm_training_data = [];
svm_test_data = [];
for q=1:cfg.n_src
    svm_training_data = [ svm_training_data;squeeze(mod_data(:,:,q))];
    svm_test_data = [ svm_test_data;squeeze(mod_test_data(:,:,q))];
end
% substract mean and 2*standard dev from each feature
feat_means = mean(svm_training_data);
feat_stds = 1./(2*std(svm_training_data));
svm_training_data = (svm_training_data - feat_means).*feat_stds;
svm_test_data = (svm_test_data - feat_means) .* feat_stds;

% permute test_data 
rand_train = randperm(length(svm_training_data));
rand_test = randperm(length(svm_test_data))';
train_label = train_label(rand_train);
test_label = test_label(rand_test);
svm_training_data = svm_training_data(rand_train,:);
svm_test_data = svm_test_data(rand_test,:);

% parameter tuning, search for best gamma and C -> grid search
bestmodel = 0;
model = [];

for log2c = -5:2:15
  for log2g = -15:2:13
    cmd = ['-h 1 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    model = svmtrain(train_label, svm_training_data, cmd);
    if (model >= bestmodel)
      bestmodel = model; 
      bestc = 2^log2c; 
      bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, model, bestc, bestg, bestmodel);
  end
end

cmd = ['-b 1 -c ', num2str(bestc), ' -g ', num2str(bestg), ];

model_lin = svmtrain(train_label, svm_training_data, '-t 0');
model = svmtrain(train_label, svm_training_data, cmd);
[predict_label, accuracy, dec_values] = svmpredict(test_label, svm_test_data, model);
[predict_label_lin, accuracy_lin, dec_values_lin] = svmpredict(test_label, svm_test_data, model_lin);

confusion_matrix_svm = zeros(cfg.n_src,cfg.n_src);
confusion_matrix_svm_lin = zeros(cfg.n_src,cfg.n_src);
for i=1:length(predict_label)
    t = test_label(i);
    p = predict_label(i);
    q = predict_label_lin(i);
    confusion_matrix_svm(p,t) = confusion_matrix_svm(p,t) +1 ;
    confusion_matrix_svm_lin(q,t) = confusion_matrix_svm_lin(q,t) +1 ;
end
figure('Name','SVM RBF KERNEL');
plotConfMat(confusion_matrix_svm, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring '});
figure('Name','SVM Linear');
plotConfMat(confusion_matrix_svm_lin, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring '});
