function [Pmu_wide,theta, freq] = wideMusicDOA(x,res,cfg)
% estimate MUSIC Spectrum for each frequency bin
%   % Input:
%   x: NxM(samples x nmics) Matrix containing the M signals captured at the sensors
%   res: angular resolution
%   cfg: config file
%   %Output;
%   Pmu_wide: Pseudo Spectrum fbins x theta
%   theta : row vector containing angles [-90:res:90]
%   freq: row vector containing frequencies

    [~,f,t] = spectrogram(x(:,1),cfg.music.window,cfg.music.n_overlap ,cfg.music.n_fft,cfg.fs);
    [N,M] = size(x);
    fbins = length(f);
    tframes = length(t);
    % perform STFT for each sensor signal 
    X = zeros(fbins,tframes,M);
    for i = 1:M
       X(:,:,i) = spectrogram(x(:,i),cfg.music.window,cfg.music.n_overlap,cfg.music.n_fft,cfg.fs);   
    end
    
    % consider only f bins specified by cfg.music.freq_range
    fmin_idx = cfg.music.freq_range_bins(1);
    fmax_idx = cfg.music.freq_range_bins(2);
    X = X(fmin_idx:fmax_idx,:,:);
    freq = cfg.music.freq(fmin_idx:fmax_idx);
    n_angles = length(-90:res:90);
    % apply narroMusicDOA for each frequency binsseparately
    Pmu_wide = zeros(length(freq),n_angles);
    for i = 1:length(freq)
        % Grab current frequency
        fi = freq(i);
        % Extract STFT information for this frequency (Rows: Time frames, Columns: Microphones) 
        xi = squeeze(X(i,:,:));  
        % Perform smallband MUSIC DOA estimation
        [Pmu_wide(i,:),theta] = narrowMusicDOA( xi, res, fi,cfg);
    end
    
end

