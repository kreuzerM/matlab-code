%edit by @M
%function mean_Diff = compute_feature(cfg,simpar)
function mean_Diff = compute_feature(array_index,cfg,signal,source_pos)

h = (rir_generator(cfg.c, cfg.fs, cfg.mic_pos(:,:,array_index), source_pos, cfg.room_dim, cfg.beta, cfg.fs*cfg.beta)).';

x = fftfilt(h,signal);
%edit by @M
x = x + sqrt(mean(var(x)) * 10^(-cfg.SNR/10)) * (randn(size(x)));
% Create the microphone signals by convolution and perform STFT-transform
%X(:,:,1) = specgram(x(:,1),cfg.n_fft,cfg.fs,cfg.window,cfg.n_overlap);
%X(:,:,2) = specgram(x(:,2),cfg.n_fft,cfg.fs,cfg.window,cfg.n_overlap);
X(:,:,1) = spectrogram(x(:,1),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
X(:,:,2) = spectrogram(x(:,2),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
%% CDR estimation
%edit cfg.lambda = cfg.nr.lambda?
%edit by @M
cfg.lambda = 0.95; % smoothing factor for PSD estimation

% estimate PSD and coherence
Pxx = estimate_psd(X,cfg.cdr.lambda);
Cxx = estimate_cpsd(X(:,:,1),X(:,:,2),cfg.lambda)./sqrt(Pxx(:,:,1).*Pxx(:,:,2));

% define coherence models
Cnn = sinc(2 * cfg.cdr.freq * cfg.d_mic/cfg.c); % diffuse noise coherence; not required for estimate_cdr_nodiffuse

% apply CDR estimator (=SNR)
CDR = estimate_cdr_nodoa(Cxx, Cnn);
CDR = max(real(CDR),0);

freq= [cfg.cdr.freq_range_bins(1):cfg.cdr.freq_range_bins(2)];
CDR_trunc = CDR(freq,:);
Diffuseness = 1./(CDR_trunc+1);
mean_Diff = mean(Diffuseness(:));

end
