function [PRPs] = PRP(Xn,cfg)
%estimate pair-wise relativce phase ratio(PRP)
% -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - PRPs: array containing PRPs for each sensor node and source (dim: 1 x narray x nsrc) 
   [fbins,tbins,nmics,nsrc,narray] = size(Xn);
   fmin_idx = cfg.music.freq_range_bins(1);
   % find index for the highest frequency that is to be considered
   fmax_idx = cfg.music.freq_range_bins(2);
   %% only look at a certain frequency interval -> shorten XTFT signal
   X = Xn(fmin_idx:fmax_idx,:,:,:,:);
   PRPs = zeros(nsrc,narray);
    for n=1:narray
        for q=1:nsrc
            Xtmp = squeeze(X(:,:,:,q,n));
            PRP = Xtmp(:,:,2).^2 ./ Xtmp(:,:,1).^2 .* (abs(Xtmp(:,:,1))./abs(Xtmp(:,:,2)));
            PRPs(q,n) = mean(mean(PRP,2));
        end
    end
end

