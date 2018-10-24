function [binMSC] = binMSC(X,cfg)
% compute binwise PRP  for a specified source and sensor node
    fmin_idx = cfg.music.freq_range_bins(1);
    fmax_idx = cfg.music.freq_range_bins(2);
    freq = cfg.cdr.freq(fmin_idx:fmax_idx);
    nfb = length(freq);
    %% only look at a certain frequency interval
    X = X(fmin_idx:fmax_idx,:,:);
    [~,ntb, ~] = size(X);
    
    cfg.lambda = 0.95; % smoothing factor for PSD estimation
    % estimate PSD and MSC
    Pxx = estimate_psd(X,cfg.cdr.lambda);
    MSC = (estimate_cpsd(X(:,:,1),X(:,:,2),cfg.lambda).^2)./(Pxx(:,:,1).*Pxx(:,:,2));
    MSC = MSC.';
    binMSC = MSC(:);
%     binMSC = zeros(nfb*ntb,1);
%     for f=1:nfb 
%         for t=1:ntb
%         offset = (f-1)*ntb;
%         binMSC(t+offset) = Cxx(f,t).^2 ./(Pxx(f,t,1)*Pxx(f,t,1));
%         end
%     end
end



