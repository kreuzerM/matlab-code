function [X] = computeSTFT(x,cfg)

%-Input: 
%   -x: time domain signal xn (dim: samples x nmics x nsrc x narray)
%   -cfg: config file
% - Output: X STFT Signal
 [~,nmics,nsrc,narray] = size(x);
    for  q = 1 : nsrc
        for n = 1:narray
                for m = 1:nmics
                    sig = squeeze(x(:,m,q,n));
                    X(:,:,m,q,n) = spectrogram(sig,cfg.music.window,cfg.music.n_overlap ,cfg.music.n_fft,cfg.fs);
                end          
        end
    end
   
end

