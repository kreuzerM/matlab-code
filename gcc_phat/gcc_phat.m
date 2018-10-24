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
            delay(q,n) = gcc_phat_single(squeeze(xn(:,1,q,n)),squeeze(xn(:,2,q,n)),cfg.fs);
        end
    end
end




