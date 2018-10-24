function [Pmu_issm,DOA] = issm_stft_single(Pmu_wide,theta)
% MUSIC_DOA_ISSM estimates a 1D spatial MUSIC pseudo spectrum based on a wideband 
% spatial MUSIC pseudo spectrum using ISSM (Incoherent Signal Subspace Method) and returns 
% the estimated DOAs of the signal sources.
%
% Input:
% Pmu_wide:    Matrix of the estimated WIDEBAND spatial MUSIC pseudo spectrum
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
        [~,ix] = max(Pmu_issm);
        DOA = theta(ix);
    
end



