function [DOA] = getDOAs(xn,cfg,mode)
% input: xn samples x nmics x nsrc x narray
[length,nmics,nsrc,narray ] = size(xn);
    n_src = cfg.n_src;
    cfg.n_src = 1;
if(mode == 1)    
    for q = 1: n_src 
        for n = 1:cfg.n_array
            [Pmu_wide(:,:,n,q),theta, freq] = wideMusicDOA(squeeze(xn(:,:,q,n)),cfg.res,cfg);
            [~,DOA(q,n)] = issmMusicDOA(Pmu_wide(:,:,n,q),theta,cfg);
        end
    end
    % dim: Pmu: fbins x tbins x n_array x n_src
    [fbins,~,~] = size(Pmu_wide); 
% calculate DOA for each frequency bin
%     for  q = 1 : n_src
%         for n = 1:cfg.n_array
%             for f = 1:fbins
%                 [~,ix] = max(Pmu_wide(f,:,n,q));
%              DOA(f,n,q) = theta(ix);
%         
% 
%             end
%         end
%     end
end
if(mode==2)
    for  q = 1 : n_src
        for n = 1:cfg.n_array
            % % DOA: bins x narray x nsrc
            sig = squeeze(xn(:,:,q,n));
                for m = 1:nmics
                    X(:,:,m) = spectrogram(sig(:,m),cfg.music.window,cfg.music.n_overlap ,cfg.music.n_fft,cfg.fs);
                end
            DOA(:,n,q) = binNarrowMusicDoa(X,cfg.res,cfg); %%   -X: STFT inputsignal:   fbins x tbins x nmic 
        end
    end
end
    cfg.n_src = n_src;
end

