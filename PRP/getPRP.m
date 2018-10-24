function [PRP] = getPRP(xn,cfg)
%estimate pair-wise relativce phase ratio(PRP)
% -Input: 
%   -xn: time domain signal xn (dim: samples x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - PRP: array containing PRPs for each sensor node and source (dim: bins x narray x nsrc)    
 [~,nmics,nsrc,narray] = size(xn); 
    for  q = 1 : nsrc
        for n = 1:narray
                for m = 1:nmics
                    sig = squeeze(xn(:,m,q,n)); %% compute STFT for each microphone signal in a loop for each node and source
                    X(:,:,m) = spectrogram(sig,cfg.music.window,cfg.music.n_overlap ,cfg.music.n_fft,cfg.fs);
                end
                PRP(:,n,q) = binPRP(X,cfg);          
        end
    end
end

