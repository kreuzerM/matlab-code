function [Pmu_issm,DOA] = issmMusicDOA(Pmu_wide,theta,cfg)
% MUSIC_DOA_ISSM estimates a 1D spatial MUSIC pseudo spectrum based on a wideband 
% spatial MUSIC pseudo spectrum using ISSM (Incoherent Signal Subspace Method) and returns 
% the estimated DOAs of the signal sources.
%
% Input:
% spec_music_wb:    Matrix of the estimated WIDEBAND spatial MUSIC pseudo spectrum
%                   --> Rows: Frequencies
%                   --> Columns: Angle theta
% theta:            angular array
% N_src:            Number of contained sources 
%
% Output:       
% spec_music_issm:  Row vector of the ISSM spatial MUSIC pseudo spectrum 
% doa:              Row vector of estimated DOAs within {-90°...+90°} [degree]
%
%
    [l_f,l_theta] = size(Pmu_wide);
   
    if(l_theta ~= length(theta))
       error('Dimension mismatch for theta-values'); 
    end    
    
    %% ISSM
   Pmu_issm  = zeros(1,l_theta);
    for i = 1:l_f
        Pmu_issm = Pmu_issm + (Pmu_wide(i,:).^(-1));
    end
    Pmu_issm = l_f.*(Pmu_issm.^(-1));

    %% Find DOAs -> find cfg.n_src largest peaks and their indices
            [~,idx] = max(Pmu_issm);
            DOA = theta(idx);
%         [Pks,locs] = findpeaks(Pmu_issm);
%          if (length(locs)== 0)
%             %error('DOA estimation failed: no peak detected')
%             [~,idx] = max(Pmu_issm);
%             DOA = theta(idx);
%          end
%         [~,idx] = sort(Pks,'descend');
%         if (length(idx) < cfg.n_src)
%             DOA = theta(locs(idx));
%             %error('DOA estimation failed: not enough peaks detected')
%         end
       % DOA = theta(locs(idx(1:cfg.n_src)));
%     elseif (N_src > 1)
%         error('Local maximum search for N_src > 1 currently not implemented');
%         % --> https://www.gomatlab.de/faq-lokale-maxima-in-einem-vektor-t6664.html    
%         %maxima = spec_music_ism(find(diff(sign(diff([0,spec_music_ism,0])))<0));
%         %index = find(diff(sign(diff([0,spec_music_ism,0])))<0);      
%     else
%         error('Choose N_src > 0')
    %end    
    
end

