function [Pmu,theta] = narrowMusicDOA( signal, res, fc,cfg)
% estimate DOA with MUSIC for a narrowband signal with center frequency fc
%  Input:
% - signal: NxM matrix containing captured microphone signals
%    - M: number of data samples
%    - N: number of sensors
% - res: angular resolution
% - fc: center frequency of impinging wave
% - cfg: config file
% Output:
% - Pmu: spatial music spectrum
% - theta: angular array
%%- DOA: estimated direction of arrival
%%  
    [N,M] = size(signal);
    theta = (-90:res:90);
    signal = signal.';
    %calculate MxM  correlation matrix:
    R = 1/N .*(signal*signal');
%%    
    % eigenvalue decomposition:
    [V,D] = eig(R);
    %% sort eigenvalues in descending order, choose eigenvectors corresponding to (n_mics-n_src) smallest eigenvalues to ...
    % form the noise sub space Qn
    [~,idx] = sort(diag(abs(D)),'descend');
    %D = D(idx,idx);
    V = V(:,idx);
    Qn = V(:,cfg.n_src+1:end); % noise sub-space
%%
    % construct the steering vector a ( dim: M x length(theta))
     a = exp(-1i*2*pi*fc*(cfg.d_mic.*sind(theta)./cfg.c).*(0:M-1).');
%%
    % compute spatial spectrum Pmu
%     Pmu = zeros(length(theta),1);
%     for k = 1:length(theta)
%         Pmu(k) = (real((a(:,k)' * ((Qn*Qn')*a(:,k))+eps))).^(-1);
%     end
    Pmu = (real(sum((((a')*(Qn*Qn')).*(a.')),2))+eps).^(-1);
    
end


