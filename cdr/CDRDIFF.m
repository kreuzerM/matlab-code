function [Diff, CDR] = CDRDIFF(Xn,cfg)
% estimate Diffuseness and Coherent-to-Diffuse-Power Ratio
% -Input: 
%   -Xn: STFT domain signal xn (dim: fbins x tbins x nmics x nsrc x narray)
%   -cfg: config file
% - Output:
%       - Diff: array containing Diffuse estimate for each sensor node and source (dim: 1 x narray x nsrc) 
%       - CDR: array containing Coherent-to-Diffuse-Power-Ration for each sensor node and source (dim: 1 x narray x nsrc) 

    [fbins,tbins,nmics,nsrc,narray] = size(Xn);
    fmin_idx = cfg.cdr.freq_range_bins(1);
    fmax_idx = cfg.cdr.freq_range_bins(2);
    X = Xn(fmin_idx:fmax_idx,:,:,:,:);
    freq = cfg.cdr.freq(fmin_idx:fmax_idx);
    Diff = zeros(nsrc,narray);
    CDR = zeros(nsrc,narray);
    for n = 1:narray
        for q = 1:nsrc  
            Xtmp = squeeze(X(:,:,:,q,n));
            Pxx = estimate_psd(Xtmp,cfg.cdr.lambda);
            Cxx = estimate_cpsd(Xtmp(:,:,1),Xtmp(:,:,2),cfg.cdr.lambda)./sqrt(Pxx(:,:,1).*Pxx(:,:,2));
            Cnn = sinc(2 * freq * cfg.d_mic/cfg.c); % diffuse noise coherence; not required for estimate_cdr_nodiffuse
            CDRtmp = estimate_cdr_nodoa(Cxx, Cnn);
            CDRtmp = max(real(CDRtmp),0);
            Diffuseness = 1./(CDRtmp+1);
            Diff(q,n) = mean(Diffuseness(:));
            CDR(q,n) = 1./Diff(q,n) -1;
        
        end
    end
end

