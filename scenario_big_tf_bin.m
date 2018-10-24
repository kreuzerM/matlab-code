clean;
config;
tic;
%% setup room, sensors and source positions
cfg.n_array = 4;
cfg.n_mic = 2;
cfg.pos_ref = [ 4 0 1.5; 0 5 1.5; 4 10 1.5; 8 5 1.5];
cfg.mic_array_rot = [ 90,0,-90,-180];
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
% generate mic positions for each array
for i=1:cfg.n_array
    cfg.mic_pos(:,:,i) = generateSensorArray(cfg.pos_ref(i,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(i));
end
%% source positions
cfg.n_src = 10;
cfg.source_pos = [2.9 1.5 1.5; 1.5 2.5 1.5; 5 1.5 1.5; 4.5 4.5 1.5;...
                 2 6.5 1.5; 6.5 6.5 1.5; 1.5 8 1.5; 6 9 1.5;
                 3.2 5.5 1.5;5.6 7 1.5];
%% source signals
cfg.sig_len = 5;
source_path_idx = [10,9,15,5,1,8,11,7,13,3];
s = zeros(cfg.fs*cfg.sig_len,cfg.n_src);
for i=1:cfg.n_src
    path = cfg.source_path{source_path_idx(i)};
    s(:,i)=getSourceSignal(path,cfg.fs,cfg.sig_len);
end

%% visualize
visualizeSetup(cfg,1);
%% generate mic signals
[xnoisy,H] = generateMicrophoneSignals(s,cfg);
X = computeSTFT(xnoisy,cfg);
% samples x nmics x nsrc x narray

%% DOA feature

DOA = getDOAs(xnoisy,cfg,2);

%% diffuseness  
[Diff,CDR] = getDiffCDR(xnoisy,cfg);


%% pairwise relative phase ratio
PRP = getPRP(xnoisy,cfg);
%% magnitude squared coherence
[MSC] = getMSC(xnoisy,cfg);
%% form data array 
  data = [];
  for q = 1:cfg.n_src
    data(:,:,q) = [squeeze(DOA(:,:,q)), squeeze(Diff(:,:,q)),squeeze(MSC(:,:,q)),squeeze(PRP(:,:,q))];
  end
%% split data into training and test data
 ndata = length(data(:,1,1));
 training_samples = floor(0.85*ndata); % number of feature vectors for training for each source
 testsamples = ndata-training_samples; % number of feature vectors for testing for each source
 for q = 1:cfg.n_src
    indices = randperm(ndata);
    training_data(:,:,q) = data(indices(1:training_samples),:,q);
    test_data(:,:,q) = data(indices(training_samples+1:end),:,q);
 end
 %% rearange test_data set
 training_data_re =[];
 maxiter = 1000;
 mu =[];
 sigma = [];
 for q = 1:cfg.n_src  
    tmp = training_data(:,:,q) ;
    %training_data_re = [training_data_re;tmp];
    [mu(q,:),tsigma] = EM(tmp,1,maxiter);
    sigma{q}= cell2mat(tsigma);
 end

%% test GMMS
confusion_matrix = zeros(cfg.n_src,cfg.n_src);
for q= 1:cfg.n_src
    tdata = squeeze(test_data(:,:,q));
    probability_matrix = zeros(cfg.n_src,length(tdata));
    for n = 1:cfg.n_src  
        probability_matrix(n,:)= gaussianND(tdata, mu(n, :), sigma{n});
    end
   [~,ix] = max(probability_matrix,[],1);
   for i=1:length(ix)
        confusion_matrix(ix(i),q) = confusion_matrix(ix(i),q) +1;
   end
end

%% plot confusion matrix
figure;
plotConfMat(confusion_matrix, {'1 coffee machine ', '2 hair dryer ',...
    '3 telephone ','4 Female3 ','5 Male1 ','6 water pouring ','7 keyboard ',...
    '8 vacuum cleaner ','9 coughing ','10 Male3 '});
toc;