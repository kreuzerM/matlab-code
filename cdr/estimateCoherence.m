function [S11, S22, S12, C] = estimateCoherence(X1,X2,lambda)
%   
%   [S11, S22, S12, C] = estimateCoherence(X1,X2,alpha)
%   Compute the auto- and cross-power spectral densities and the complex coherence.
%   X1 and X2 are signals in the short-time Fourier domain
%   lambda is the forgetting factor for the recursive averaging (default: 0.68)
%   compute the power spectral density with recursive averaging


if (nargin == 2)
    lambda = 0.68;
end

S11 = single(estimate_psd(X1,lambda));
S22 = single(estimate_psd(X2,lambda));
S12 = single(estimate_cpsd(X1,X2,lambda));

C = S12./sqrt(S11.*S22);

C(C>1) = 1*exp(1j*angle(C(C>1)));

end
