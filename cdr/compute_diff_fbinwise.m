function [meanDiffbin,meanCDRbin] = compute_diff_fbinwise(x,cfg)
%function mean_Diff = compute_feature(array_index,cfg,signal,source_pos)
% -input: x: samples x nmics
    [n,m] = size(x);
    if(m ~=2)
        error('wrong number of microphones - not a microphone pair');
    end
    X(:,:,1) = spectrogram(x(:,1),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
    X(:,:,2) = spectrogram(x(:,2),cfg.cdr.window,cfg.cdr.n_overlap ,cfg.cdr.n_fft,cfg.fs);
   
    %% estimate CDR for each frequency bin
    %% only evaluate certain frequency bins
    
    fmin_idx = cfg.cdr.freq_range_bins(1);
    fmax_idx = cfg.cdr.freq_range_bins(2);
    X = X(fmin_idx:fmax_idx,:,:);
    freq = cfg.cdr.freq(fmin_idx:fmax_idx);
    nfb = length(freq);
    
    meanDiffbin = zeros(nfb,1);
    meanCDRbin = zeros(nfb,1);
    
    
    
    cfg.lambda = 0.95; % smoothing factor for PSD estimation
    for f=1:nfb
    % CDR estimation
        sig =  squeeze(X(f,:,:));
    % estimate PSD and coherence
        Pxx = estimate_psd(sig,cfg.cdr.lambda);
        Cxx = estimate_cpsd(sig(:,1),sig(:,2),cfg.lambda)./sqrt(Pxx(:,1).*Pxx(:,2));

        % define coherence models
        %fc = cfg.cdr.freq_range_bins(1)+f;
        fc = freq(f);
        Cnn = sinc(2 * fc * cfg.d_mic/cfg.c); % diffuse noise coherence; not required for estimate_cdr_nodiffuse

        % apply CDR estimator (=SNR)
        CDR = estimate_cdr_nodoa(Cxx, Cnn);
        CDR = max(real(CDR),0);

       
        Diffuseness = 1./(CDR+1);
        meanDiffbin(f) = mean(Diffuseness(:));
        meanCDRbin(f) = 1./meanDiffbin(f) -1;
    end
end

