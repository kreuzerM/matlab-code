function [DOA] = wideMusicDOA_STFT_domain(X,freq,res,cfg)
% estimate MUSIC Spectrum for each frequency bin
%   % Input:
%   X: fbins x tbins x nmics  Matrix containing M signals captured at the sensors
%   res: angular resolution
%   cfg: freq vector containing the center frequencies for each frequency
%   bin
%   %Output;
%   Pmu_wide: Pseudo Spectrum fbins x theta
%   theta : row vector containing angles [-90:res:90]
%   freq: row vector containing frequencies   
    n_angles = length(-90:res:90);
    % apply narroMusicDOA for each frequency bins separately
    Pmu_wide = zeros(length(freq),n_angles);
    for i = 1:length(freq)
        % Grab current frequency
        fi = freq(i);
        % Extract STFT information for this frequency (Rows: Time frames, Columns: Microphones) 
        xi = squeeze(X(i,:,:));  
        % Perform smallband MUSIC DOA estimation
        [Pmu_wide(i,:),theta] = narrowMusicDOA( xi, res, fi,cfg);
    end
    [~,DOA] = issm_stft_single(Pmu_wide,theta);
end

