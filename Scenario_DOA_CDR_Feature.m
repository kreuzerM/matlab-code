%% clean workspace load config
clean;
config;
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');  

%% define geometric setup
cfg.n_src = 3;
cfg.source_pos = [ 3 4.5 1.5 ;4.7 5.5 1.5; 4.1 6 1.5];
% reference point and rotation for each array
cfg.n_array = 4;
cfg.n_mic = 2;
cfg.pos_ref = [ 4 3 1.5; 2 5 1.5; 4 7 1.5; 6 5 1.5];
cfg.mic_array_rot = [ 90,0,-90,-180];
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
% generate mic positions for each array
for i=1:cfg.n_array
    cfg.mic_pos(:,:,i) = generateSensorArray(cfg.pos_ref(i,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(i));
end


%% load and shorten source signals
cfg.sig_len = 5;
[tmp,fs] = audioread(cfg.source_path{1});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s1 = tmp;
var1 = var(s1,1);

[tmp,fs] = audioread(cfg.source_path{5});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s2 = tmp;
var2= var(s2,1);
s2 = s2* sqrt(var1/var2);

[tmp,fs] = audioread(cfg.source_path{3});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s3 = tmp;
var3= var(s3,1);
s3 = s3* sqrt(var1/var3);

s = [s1,s2,s3]; 
%% generate RIRs for each sensor array
    % dim(H) = nsamples x nsrc x nmic x narray
for i = 1:cfg.n_array
   H(:,:,:,i) = wrap_rir_generator(i,cfg);
end

%% convolute RIRs with source signals
for i = 1:cfg.n_array
    for q = 1:cfg.n_src
        x(:,:,q,i) = fftfilt(squeeze(H(:,q,:,i)),s(:,q)); % dim(x) samples x nmics x nsrc x narray   
    end
end
%% generate noisy microphone signals for each sensor array
xnoise = 2*rand(size(x(:,:,:,1)))-1;
xnoise = squeeze(xnoise);% same noise sequence for all signals
for i = 1:cfg.n_array
    xtmp = x;
    %xtmp = sum(xtmp,3);
    scalefac = max(max(abs(xtmp)));
    xtmp = xtmp./scalefac;
    xn(:,:,:,i) = addfixedNoise(squeeze(xtmp(:,:,:,i)),xnoise,cfg.SNR); % size xn: samples x nmics x nsrc x narray
end


%% pseudo music spectrum for each sensor array
cfg.n_src = 1;
for i = 1:cfg.n_array
    [Pmu_wide1(:,:,i),theta, freq] = wideMusicDOA(squeeze(xn(:,:,1,i)),cfg.res,cfg);
    [Pmu_wide2(:,:,i),theta, freq] = wideMusicDOA(squeeze(xn(:,:,2,i)),cfg.res,cfg);
    [Pmu_wide3(:,:,i),theta, freq] = wideMusicDOA(squeeze(xn(:,:,3,i)),cfg.res,cfg);
    [meanDiffbin1(:,i),meanCDRbin1(:,i)] = compute_diff_fbinwise(squeeze(xn(:,:,1,i)),cfg);
    [meanDiffbin2(:,i),meanCDRbin2(:,i)] = compute_diff_fbinwise(squeeze(xn(:,:,2,i)),cfg);
    [meanDiffbin3(:,i),meanCDRbin3(:,i)] = compute_diff_fbinwise(squeeze(xn(:,:,3,i)),cfg);
end

[fbins,~,~] = size(Pmu_wide1); 
%% calculate DOA for each bin
DOA1 = size(cfg.n_array, fbins);
DOA2 = size(cfg.n_array, fbins);
DOA3 = size(cfg.n_array, fbins);
 for i = 1:cfg.n_array
    for f = 1:fbins
        [~,ix1] = max(Pmu_wide1(f,:,i));
        [~,ix2] = max(Pmu_wide2(f,:,i));
        [~,ix3] = max(Pmu_wide3(f,:,i));
         DOA1(i,f) = theta(ix1);
         DOA2(i,f) = theta(ix2);
         DOA3(i,f) = theta(ix3);
       
    end
 end

 
%% show estimated points
cfg.n_src = 3;
fig1 = visualizeSetup(cfg,0);

%
data = [DOA1.',meanDiffbin1; DOA2.',meanDiffbin2];
data3 = [DOA3.',meanDiffbin3];
%%
k=2;
maxiter=1000;
[mu,sigma] = EM(data,k,maxiter);
%% visualize Results
fig2 = visualizeSetup(cfg,0);
figure(fig2);
%scatter(data(:,1),data(:,2),'.g')
scatter(mu(:,1), mu(:,2), 'x');
%%
%% color the clusters
p1 = gaussianND(data, mu(1, :), sigma{1});
p2 = gaussianND(data, mu(2, :), sigma{2});
p3_1 = gaussianND(data3, mu(1, :), sigma{1});
p3_2 = gaussianND(data3, mu(2, :), sigma{2});
cluster1 = data( find(p1 > p2),:);
cluster2 = data(find(p1 < p2),:);
data3_1 = data3( find(p3_1 > p3_2),:);
data3_2 = data3( find(p3_1 < p3_2),:);
points_in_cluster1 = length(data3_1(:,1))
points_in_cluster2 = length(data3_2(:,1))
