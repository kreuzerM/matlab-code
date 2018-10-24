function [source_pos] = generateSourcePosRef(ref,rho,theta,array_rot)
% generate source positions that are in a distance of rho[m] and an angle
% theta to the position ref 
% only x-y plane considered for distance

    if(length(rho)~= length(theta))
        error('dimension mismatch between rho and theta')
    end
    theta = theta+array_rot;
    tmp = zeros(length(rho),3);
    tmp(:,1) = rho;
    for i=1:length(theta)
        tmp(i,:) = rotate(tmp(i,:),theta(i)); 
    end
    
    source_pos = ref+tmp;
    
end

