function [ delay ] = gcc_phat( xn,cfg )
%estimate time delay between microphone signals with generalized cross
%correlation
% -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - delay: array containing time delay for each sensor node and source (dim: 1 x narray x nsrc)
    [samples,nmics,nsrc,narray] = size(xn);
    for n = 1:narray
        for q = 1:nsrc  
            p = floor(192e3/cfg.fs);
            x1 = squeeze(xn(:,1,q,n));
            x2 = squeeze(xn(:,2,q,n));
            x1 = resample(x1,p,1);
            x2 = resample(x2,p,1);
            delay(q,n) = gcc_phat_single(x1,x2,(p*cfg.fs));
        end
    end
end




