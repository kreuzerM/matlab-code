%% clean workspace load config
clean;
config;

%% define geometric setup
%cfg.music.freq_range = [300,4000]; 
cfg.music.freq_range = [700,1000];
cfg.n_src = 2;
cfg.source_pos = [ 3 4.5 1.5 ;4.5 5.25 1.5];
% reference point and rotation for each array
cfg.n_array = 4;
cfg.n_mic = 2;
cfg.pos_ref = [ 4 4 1.5; 2 5 1.5; 4 6 1.5; 6 5 1.5];
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

s = [s1,s2]; 
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
xnoise = 2*rand(size(x(:,:,1,1)))-1; % same noise sequence for all signals
xnoise = squeeze(xnoise);
for i = 1:cfg.n_array
    xtmp = x;
    xtmp = sum(xtmp,3);
    xtmp = squeeze(xtmp); % samples x mics x arrays
    scalefac = max(max(abs(xtmp)));
    xtmp = xtmp./scalefac;
    xn(:,:,i) = addfixedNoise(squeeze(xtmp(:,:,i)),xnoise,cfg.SNR); % size xn: samples x nmics  x narray
end


%% pseudo music spectrum for each sensor array

for i = 1:cfg.n_array
    for mic= 1:cfg.n_mic
       [X(:,:,mic,i),fbins,tbins] = spectrogram(xn(:,mic,i),cfg.music.window,cfg.music.n_overlap,cfg.music.n_fft,cfg.fs);   
    end
end
cfg.n_src = 1;

for i = 1:cfg.n_array
    DOA(i,:) = binNarrowMusicDoa(squeeze(X(:,:,:,i)),cfg.res,cfg);
end
% size Pmu_wide1/2 : fbins x theta x n_array
[~,bins] = size(DOA); 
%% calculate DOA for each bin


 %% estimate Position
    data = zeros(bins,2);
   
    for f=1:bins
        alpha = DOA(:,f) + cfg.mic_array_rot.' ;        
        A = [sind(alpha),-cosd(alpha)];
        b = [cfg.pos_ref(:,1) .* sind(alpha) - cfg.pos_ref(:,2) .*cosd(alpha)];
        data(f,:) = pinv(A)*b;
             
    end
%% show estimated points
cfg.n_src = 2;
fig1 = visualizeSetup(cfg,1);
figure(fig1);
scatter(data(:,1),data(:,2),'.g')
%% calculate mean and var and pdf  of the data sets
Sigma = cov(data);
%%  Choose initial values for the parameters.
% Set 'm' to the number of data points.
m = size(data, 1);

k = 2;  % The number of clusters.
n = 2;  % The vector lengths.

% Randomly select k data points to serve as the initial means.
indeces = randperm(m);
mu = data(indeces(1:k), :);

sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
for (j = 1 : k)
    sigma{j} = Sigma;
end

% Assign equal prior probabilities to each cluster.
phi = ones(1, k) * (1 / k);
%% Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
W = zeros(m, k);

% Loop until convergence.
for iter = 1:5000
    
    fprintf('  EM Iteration %d\n', iter);

    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k);
    
    % For each cluster...
    for j = 1:k
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(data, mu(j, :), sigma{j});
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi);
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
    
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.

    % Store the previous means.
    prevMu = mu;    
    
    % For each of the clusters...
    for j = 1 : k
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1);
        
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        mu(j, :) = weightedAverage(W(:, j), data);

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        sigma_k = zeros(n, n);
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, data, mu(j, :));
        
        % Calculate the contribution of each training example to the covariance matrix.
        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        
        % Divide by the sum of weights.
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
    % Check for convergence.
    if (mu == prevMu)
        break
    end
            
% End of Expectation Maximization    
end
%% visualize Results
fig2 = visualizeSetup(cfg,1);
figure(fig2);
scatter(data(:,1),data(:,2),'.g')
scatter(mu(:,1), mu(:,2), 'x');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 300;
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
axis([0 cfg.room_dim(1) 0 cfg.room_dim(2)]);

title('Original Data and Estimated PDFs');
