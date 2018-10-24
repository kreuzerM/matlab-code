 function [Diffbin,CDRbin] = compute_diff_binwise(x,cfg)
%function mean_Diff = compute_feature(array_index,cfg,signal,source_pos)
% % -input: x: samples x nmics
    [n,m] = size(x);
    if(m ~=2)
        error('wrong number of microphones - not a microphone pair');
    end
    X(:,:,1) = spectrogram(x(:,1),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
    X(:,:,2) = spectrogram(x(:,2),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
   
    
    %% estimate CDR for each frequency bin
    %% only evaluate certain frequency bins
    [~,ntb, ~] = size(X);
    % consider only the f bins specified in the frequency interval cfg.music.freq_range
    fmin_idx = cfg.music.freq_range_bins(1);
    fmax_idx = cfg.music.freq_range_bins(2);
    freq = cfg.cdr.freq(fmin_idx:fmax_idx);
    nfb = length(freq);
    %% only look at a certain frequency interval
    X = X(fmin_idx:fmax_idx,:,:);
    cfg.lambda = 0.95; % smoothing factor for PSD estimation
    % estimate PSD and coherence
    Pxx = estimate_psd(X,cfg.cdr.lambda);
    Cxx = estimate_cpsd(X(:,:,1),X(:,:,2),cfg.lambda)./sqrt(Pxx(:,:,1).*Pxx(:,:,2));

    % define coherence models
    Cnn = sinc(2 * freq* cfg.d_mic/cfg.c); % diffuse noise coherence; not required for estimate_cdr_nodiffuse
    
    %%
    
    Diffbin = zeros(nfb*ntb,1);
    CDRbin = zeros(nfb*ntb,1);
    cfg.lambda = 0.95; % smoothing factor for PSD estimation
    for f=1:nfb 
        for t=1:ntb
        % apply CDR estimator (=SNR)
        CDR = estimate_cdr_nodoa(Cxx(f,t), Cnn(f));
        CDR = max(real(CDR),0);

        offset = (f-1)*ntb;
        Diffuseness = 1./(CDR+1);
        Diffbin(t+offset) = Diffuseness;
        CDRbin(t+offset) = CDR;
        end
    end
end



