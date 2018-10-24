function MSCs = MSC(Xn,cfg)
%estimate magnitude squared coherence
% -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output: 
%       - MSCs: array containing MSRc for each sensor node and source (dim: 1 x narray x nsrc) 

  [fbins,tbins,nmics,nsrc,narray] = size(Xn);
   fmin_idx = cfg.cdr.freq_range_bins(1);
   % find index for the highest frequency that is to be considered
   fmax_idx = cfg.cdr.freq_range_bins(2);
   % vector containing all frequencies that have to be evaluated
   %% only look at a certain frequency interval -> shorten XTFT signal
   X = Xn(fmin_idx:fmax_idx,:,:,:,:);
 
   MSCs = zeros(nsrc,narray);
    for n = 1:narray
        for q = 1:nsrc   
            % estimate PSD and MSC
            Xtmp = squeeze(X(:,:,:,q,n));
            Pxx = estimate_psd(Xtmp,cfg.cdr.lambda);
            MSC = (abs(estimate_cpsd(Xtmp(:,:,1),Xtmp(:,:,2),cfg.cdr.lambda).^2))./(Pxx(:,:,1).*Pxx(:,:,2));
            MSCs(q,n) = mean(mean(MSC,2));
            
            
        end
    end
end

