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
% generate mic positions for each node
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
% visualizeSetup(cfg,1);
%% training & test source signals 
cfg.sig_len = 3;
% source_paths:
% (1) Male1 (2) Male1 (3)Male3 (4) Female1 (5) Female3 (6) Female 4 
% (7) vaccum cleaner (8) water pouring (9) hair dryer (10) coffee machine
% (11) keyboard (12) water stir (13) male coughinh (14) male snoring
% (15) telephone ring
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
train_rtf = [];
test_rtf = [];
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
    train_i = i
    % introduce some variance to source positions
    %offset = 0.4 +(2*randn(cfg.n_src,2)-1);
    offset = 0.5 *(2*rand(cfg.n_src,2)-1);
    cfg.source_pos(:,1:2) = true_pos(:,1:2) + offset;
    % vary the SNR and T60
    SNR = [-5,0,5,10,15,20,25,30];
    T60 = 1;
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
    train_rtf{i}  = estimateRTF( Xnoisy,cfg );
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
    test_i = i
    %choose random SNR and T60
    offset = 0.5 *(2*rand(cfg.n_src,2)-1);
    cfg.source_pos(:,1:2) = true_pos(:,1:2) + offset;
    SNR = [-5,0,5,10,15,20,25,30];
    %SNR = [18,25,22];
    T60 = [1];
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
    test_rtf{i}  = estimateRTF( Xnoisy,cfg );
end
run_time = toc(tstart);
test_snr = repmat(test_snr,cfg.n_src,1);
test_t60 = repmat(test_t60,cfg.n_src,1);
%test_pos = repmat(test_pos_offset,number_test_data,1);
training_snr = repmat(training_snr,cfg.n_src,1);
training_t60 = repmat(training_t60,cfg.n_src,1);
%training_pos = repmat(tra_pos_offset,number_training_data,1);
%% evaluation part
% select only certain features
feat_idx{1} = [1:4];
feat_idx{2} = [5:8];
feat_idx{3} = [9:12];
feat_idx{4} = [13:16];
feat_idx{5} = [17:24];
feat_idx{6} = [25:28];
feat_idx{7} = [1:8];
acc_complete=[];
pred_complete=[];

%for p = 1:7
feat_vec = feat_idx{1};
mod_data = data(:,feat_vec,:);
mod_test_data = test_data(:,feat_vec,:);
 test_data_comp = [];
  training_data_comp =[];
for q=1:cfg.n_src
    training_data_comp = [ training_data_comp;squeeze(mod_data(:,:,q))];
    test_data_comp = [test_data_comp;squeeze(mod_test_data(:,:,q))];
end


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
test_label =  ones(length(test_data),1) * [1:cfg.n_src];
train_label = train_label(:);
test_label = test_label(:);
svm_training_data = [];
svm_test_data = [];
for q=1:cfg.n_src
    svm_training_data = [ svm_training_data;squeeze(mod_data(:,:,q))];
    svm_test_data = [ svm_test_data;squeeze(mod_test_data(:,:,q))];
end


% soft scaling
% substract mean and 2*standard dev from each feature
feat_means = mean(svm_training_data);
feat_stds = 1./(2*std(svm_training_data)+eps);
svm_training_data = (svm_training_data - feat_means).*feat_stds;
svm_test_data = (svm_test_data - feat_means) .* feat_stds;

% % hard scaling
% column_min = min(svm_training_data);
% column_max = max(svm_training_data);
% svm_training_data = rescale(svm_training_data,'InputMin',column_min,'InputMax',column_max);
% svm_test_data = rescale(svm_test_data,'InputMin',column_min,'InputMax',column_max);

% % permute test_data 
% rand_train = randperm(length(svm_training_data));
% rand_test = randperm(length(svm_test_data))';
% train_label = train_label(rand_train);
% test_label = test_label(rand_test);
% svm_training_data = svm_training_data(rand_train,:);
% svm_test_data = svm_test_data(rand_test,:);


% confusion_matrix_svm = zeros(cfg.n_src,cfg.n_src);
% confusion_matrix_svm_lin = zeros(cfg.n_src,cfg.n_src);
% for i=1:length(pred_svm_rbf)
%     t = test_label(i);
%     p = pred_svm_rbf(i);
%     q = pred_svm_lin(i);
%     confusion_matrix_svm(p,t) = confusion_matrix_svm(p,t) +1 ;
%     confusion_matrix_svm_lin(q,t) = confusion_matrix_svm_lin(q,t) +1 ;
% end
% figure('Name','SVM RBF KERNEL');
% plotConfMat(confusion_matrix_svm, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring '});
% figure('Name','SVM Linear');
% plotConfMat(confusion_matrix_svm_lin, {'1 coffee machine ', '2 hair dryer ', '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring '});

% cm =confusionmat(test_label,pred);
% figure('Name','Confusion Matrix');
% plotConfmat(cm);

%[pred_lp, acc_lp ] = gp_laplace( train_label, test_label,svm_training_data, svm_test_data );
%[pred_ep, acc_ep ] = gp_ep( train_label, test_label,svm_training_data, svm_test_data );
[pred_svm_rbf, acc_svm_rbf] = svm_rbf( train_label, test_label,svm_training_data, svm_test_data );
[pred_svm_lin, acc_svm_lin ] = svm_lin( train_label, test_label,svm_training_data, svm_test_data );
[pred_svm_1vsR, acc_svm_1vsR ] = svm_rbf_one_vs_all( train_label, test_label,svm_training_data, svm_test_data );
[pred_gp, acc_gp ] = gp_classifier(train_label, test_label,svm_training_data, svm_test_data );
acc_gp
[pred_gmm, acc_gmm ] = gmm( train_label, test_label,svm_training_data, svm_test_data );
[pred_gmm2, acc_gmm2 ] = gmm( train_label, test_label,training_data_comp, test_data_comp );
[pred_gp1vs_all, acc_gp_1vs_all] =gp_classifier_1_vs_all(train_label, test_label,svm_training_data, svm_test_data);

acc = [acc_gmm,acc_svm_lin(1)/100,acc_svm_rbf(1)/100,acc_svm_1vsR,acc_gp,acc_gp_1vs_all];
pred = [pred_gmm,pred_svm_lin,pred_svm_rbf,pred_svm_1vsR,pred_gp,pred_gp1vs_all];

for k=1:length(acc)
 idx1 = find(test_snr >=  5);
 idx2 = find(test_snr <= 0);
 
 acc_subset_high_snr(k) = sum(test_label(idx1) == pred(idx1,k))./ numel(test_label(idx1));
 acc_subset_low_snr(k) = sum(test_label(idx2) == pred(idx2,k))./ numel(test_label(idx2));
end
 acc_complete{p}=[acc;acc_subset_low_snr;acc_subset_high_snr].'
 pred_complete{p} = pred;
%end
% %% visualize with predited label
% miss_idx = find(test_label ~= pred(:,5));
% cfg.source_pos = true_pos;
% setup = visualizeSetup(cfg,1);
% %colormap([ 1 0 0 ; 1 0.5 1; 1 1 0;  0 1 0; 0 0 1; 0.5 0 1]);
% colormap(parula(8))
% cb=colorbar;
% %set(cb,'YTick',[1:max(test_label)])
% set(cb,'YTick',[-5:5:30])
% %cb.Label.String = 'Class label index';
% cb.Label.String = 'SNR[dB]';
% test_pos_c = [test_pos_s1;test_pos_s2;test_pos_s3;test_pos_s4;test_pos_s5;test_pos_s6];
% scatter(test_pos_c(miss_idx,1),test_pos_c(miss_idx,2),10,test_snr(miss_idx));