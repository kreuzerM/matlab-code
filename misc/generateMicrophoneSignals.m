function [xnoisy,H] = generateMicrophoneSignals(s,cfg)
% generate noise microphone signals by convoluting source signals s with
% RIRs and adding white gaussian noise
% input:
   %s: dim: cfg.fs*cfg.sig_len x n_src
   %cfg: config file
 % output:
  
    % xnoisy: samples x nmics x nsrc x narray 
     %% Generate RIRs
%     for i = 1:cfg.n_array
%         H(:,:,:,i) = wrap_rir_generator(i,cfg);
%     end
 %%
 %% generate noise sequence
    xnoise = 2*rand(cfg.fs*cfg.sig_len,cfg.n_mic)-1;
    
    %% convolute RIRs with source signals
    for i = 1:cfg.n_array
        H(:,:,:,i) = wrap_rir_generator(i,cfg);
        for q = 1:cfg.n_src
            x(:,:,q,i) = fftfilt(squeeze(H(:,q,:,i)),s(:,q)); % dim(x) samples x nmics x nsrc x narray 
            xtmp = squeeze(x(:,:,q,i));
            scalefac = max(max(abs(xtmp)));
            xtmp = xtmp./scalefac;
            xn(:,:,q,i) = addfixedNoise(xtmp,xnoise,cfg.SNR);
        end
    end
   
%     for n = 1:cfg.n_array
%         for q = 1:cfg.n_src
%             xtmp = squeeze(x(:,:,q,n));
%             scalefac = max(max(abs(xtmp)));
%             xtmp = xtmp./scalefac;
%             xn(:,:,q,n) = addfixedNoise(xtmp,xnoise,cfg.SNR);
%         end
%     end
    xnoisy= xn;
end

