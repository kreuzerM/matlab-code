function [DOAs] = DOA(Xn,cfg)
%estimate DOAs
% -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output:
%       - DOA: array containing DOA estimates for each sensor node and source (dim: 1 x narray x nsrc) 
    [fbins,tbins,nmics,nsrc,narray] = size(Xn);
    fmin_idx = cfg.music.freq_range_bins(1);
    fmax_idx = cfg.music.freq_range_bins(2);
    X = Xn(fmin_idx:fmax_idx,:,:,:,:);
    freq = cfg.music.freq(fmin_idx:fmax_idx);
    DOAs = zeros(nsrc,narray);
    for n = 1:narray
        for q = 1:nsrc  
            Xtmp = squeeze(X(:,:,:,q,n));
            % Xtmp : fbins x tbins x nmics
            DOAs(q,n) = wideMusicDOA_STFT_domain(Xtmp,freq,cfg.res,cfg);
        
        end
    end

end

