function [DOA] = binNarrowMusicDoa(X,res,cfg)
%binwise DOA estimation for a microphone array
%  input:
%   -X: STFT inputsignal:   fbins x tbins x nmic 
%   -cfg: config file
% output:
%   - DOA: row vector containing estimated DOA for each bin(1xfbins*tbins )
    [~,ntb, ~] = size(X);
    % consider only the f bins specified in the frequency interval cfg.music.freq_range
    fmin_idx = cfg.music.freq_range_bins(1);
    fmax_idx = cfg.music.freq_range_bins(2);
    X = X(fmin_idx:fmax_idx,:,:);
    freq = cfg.music.freq(fmin_idx:fmax_idx);
    nfb = length(freq);
    
    DOA = [zeros(1,nfb*ntb)];
    for f=1:nfb
        fc = freq(f);
        for t=1:ntb
             sig =squeeze(X(f,t,:));
             sig = sig.';
            [Pmu,theta] = narrowMusicDOA( sig, res, fc,cfg);
            offset = (f-1)*ntb;
            [~,ix] = max(Pmu);
            DOA(t+offset) = theta(ix);
        end
    end
end

