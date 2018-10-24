function [xn] = addNoise(x,snr)
% add uncorrelated white noise to input signal x for the specified snr
xnoise = 2*rand(size(x))-1;
% adjust SNR to desired value 
scalefac = min(sqrt(mean(sum(x.^2,1))./mean(sum(xnoise.^2,1))./10.^(snr/10)));
xnoise = xnoise.*scalefac; 
xn = x + xnoise;

end

