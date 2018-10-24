function [binPRP] = binPRP(X,cfg)
% compute binwise PRP  for a specified source and sensor node
% -Input: 
%   -X: STFT Signal of a microphone pair(dim: fbins x tbins x 2)
%   -cfg: config file
% - Output:
%   -binPRP: binwise PRP for a sensor node and a source(row vector of ...
%            length fbins x tbins

    % find index for the lowest frequency that is to be considered
    fmin_idx = cfg.music.freq_range_bins(1);
    % find index for the highest frequency that is to be considered
    fmax_idx = cfg.music.freq_range_bins(2);
    % vector containing all frequencies that have to be evaluated
    freq = cfg.cdr.freq(fmin_idx:fmax_idx);
    % number of frequency bins
    nfb = length(freq);
    %% only look at a certain frequency interval -> shorten XTFT signal
    X = X(fmin_idx:fmax_idx,:,:);
    [~,ntb, ~] = size(X);
    binPRP = zeros(nfb*ntb,1);
    PRP = X(:,:,2).^2 ./ X(:,:,1).^2 .* (abs(X(:,:,1))./abs(X(:,:,2)));
    PRP = PRP.';
    binPRP = PRP(:);
    % PRP value correspoding to bin X(2,1) is binPRP(ntb+1); 
    
%     for f=1:nfb 
%         for t=1:ntb
%             offset = (f-1)*ntb; % counting offset
%             binPRP(t+offset) = X(f,t,2)/X(f,t,1) * (abs(X(f,t,1))/abs(X(f,t,2))) ;
%            
%         end
%     end
    
    

end

