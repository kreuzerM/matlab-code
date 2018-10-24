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
%cfg.pos_ref = [ 4 3 1.5; 2 5 1.5; 4 7 1.5; 6 5 1.5];
cfg.pos_ref = [ 4 0.5 1.5; 0.5 5 1.5; 4 9.5 1.5; 7.5 5 1.5];
cfg.mic_array_rot = [ 90,0,-90,-180];
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
% generate mic positions for each array
for i=1:cfg.n_array
    cfg.mic_pos(:,:,i) = generateSensorArray(cfg.pos_ref(i,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(i));
end


%% load and shorten source signals
cfg.sig_len = 5;
[tmp,fs] = audioread(cfg.source_path{7});

tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s1 = tmp;
var1 = var(s1,1);

[tmp,fs] = audioread(cfg.source_path{8});
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

 %% estimate Position
    data1 = zeros(fbins,2);
    data2 = zeros(fbins,2);
    data3 = zeros(fbins,2);
    for f=1:fbins
        alpha1 = DOA1(:,f) + cfg.mic_array_rot.' ;
        alpha2 = DOA2(:,f) + cfg.mic_array_rot.' ;
        alpha3 = DOA3(:,f) + cfg.mic_array_rot.' ;
        A1 = [sind(alpha1),-cosd(alpha1)];
        A2 = [sind(alpha2),-cosd(alpha2)];
        A3 = [sind(alpha3),-cosd(alpha3)];
        b1 = [cfg.pos_ref(:,1) .* sind(alpha1) - cfg.pos_ref(:,2) .*cosd(alpha1)];
        b2 = [cfg.pos_ref(:,1) .* sind(alpha2) - cfg.pos_ref(:,2) .*cosd(alpha2)];
        b3 = [cfg.pos_ref(:,1) .* sind(alpha3) - cfg.pos_ref(:,2) .*cosd(alpha3)];
        data1(f,:) = pinv(A1)*b1;
        data2(f,:) = pinv(A2)*b2; 
        data3(f,:) = pinv(A3)*b3;  
    end
    data = [data1;data2];
%% show estimated points
cfg.n_src = 3;
fig1 = visualizeSetup(cfg,0);
figure(fig1);
%scatter(data1(:,1),data1(:,2),'.b');
%scatter(data2(:,1),data2(:,2),'.g');

scatter(data(:,1),data(:,2),'.g')
scatter(data3(:,1),data3(:,2),'vg')
%% calculate mean and var and pdf  of the data sets
% mu1 =  [mean(data1(:,1)), mean(data1(:,2))];
% mu2 = [mean(data2(:,1)), mean(data2(:,2))];
% sigma1 = cov(data1);
% sigma2 = cov(data2);   
% P1 = gaussianND(data1, mu1, sigma1);
% P2 = gaussianND(data2, mu2, sigma2);
% Sigma = cov(data);
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
scatter(cluster1(:,1),cluster1(:,2),'.r');
scatter(cluster2(:,1),cluster2(:,2),'.b');
data3_1 = data3( find(p3_1 > p3_2),:);
data3_2 = data3( find(p3_1 < p3_2),:);
scatter(data3_1(:,1),data3_1(:,2),'vm');
scatter(data3_2(:,1),data3_2(:,2),'vc');
%%
% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 1000;
u = linspace(0, cfg.room_dim(1), gridSize);
v = linspace(0, cfg.room_dim(2), gridSize);
[A B] = meshgrid(u, v);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussianND(gridX, mu(1, :), sigma{1});
z2 = gaussianND(gridX, mu(2, :), sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, v, Z1);
[C, h] = contour(u, v, Z2);


title('Original Data and Estimated PDFs');


