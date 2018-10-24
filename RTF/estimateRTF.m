function [ RTFs ] = estimateRTF( Xn,cfg )
% estimate RTF like it is described in http://www.eng.biu.ac.il/gannot/files/2012/05/07906609.pdf
%  -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - RTFs: array containing RTFs for each sensor node and source (dim: samps x narray x nsrc) 
[fbins,tbins,nmics,nsrc,narray] = size(Xn);
   fmin_idx = cfg.cdr.freq_range_bins(1);
   % find index for the highest frequency that is to be considered
   fmax_idx = cfg.cdr.freq_range_bins(2);
   %% only look at a certain frequency interval -> shorten XTFT signal
   X = Xn(fmin_idx:fmax_idx,:,:,:,:);
   [fbins,tbins,nmics,nsrc,narray] = size(X);
   RTFs = zeros(tbins*fbins,nsrc,narray);
    for n=1:narray
        for q=1:nsrc
            X1 = squeeze(X(:,:,1,q,n));
            X2 = squeeze(X(:,:,2,q,n));
            S_12 = estimate_cpsd(X1,X2,cfg.cdr.lambda);
            S_11 = estimate_psd(X1,cfg.cdr.lambda);
            S = S_12 ./ S_11;
            RTFs(:,q,n) = S(:);
        end
    end

end

