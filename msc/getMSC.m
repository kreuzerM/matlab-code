function [MSC] = getMSC(xn,cfg)
% estimate Magnitude squared coherence(MSC) for a time domain signal

% -Input: 
%   -xn: time domain signal xn (dim: samples x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - MSC: array containing MSC for each sensor node and source (dim: bins x narray x nsrc)    

    [samples,nmics,nsrc,narray] = size(xn);
    % xn:samples x nmics x nsrc x narray
    for  q = 1 : nsrc
        for n = 1:narray
                for m = 1:nmics
                    sig = squeeze(xn(:,m,q,n));
                    X(:,:,m) = spectrogram(sig,cfg.music.window,cfg.music.n_overlap ,cfg.music.n_fft,cfg.fs);
                end
                MSC(:,n,q) = binMSC(X,cfg);          
        end
    end
end



