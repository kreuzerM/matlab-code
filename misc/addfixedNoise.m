function [xn] = addfixedNoise(x,xnoise,snr)
% add uncorrelated white noise to input signal x for the specified snr for
% the given noise sequence xnoise


% adjust SNR to desired value 
scalefac = min(sqrt(mean(sum(x.^2,1))./mean(sum(xnoise.^2,1))./10.^(snr/10)));
xnoise = xnoise.*scalefac; 
xn = x + xnoise;
%SNR_new = 10*log10(var(x,1)./var(xnoise,1))

end

