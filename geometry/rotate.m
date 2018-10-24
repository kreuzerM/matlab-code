function [vR] = rotate(v,theta)
%rotate vector theta
%   
    
    R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
    vR = zeros(size(v));
    for i=1:length(v(:,1))
        tmp = v(i,:).';
    vR(i,:) = [(R*tmp(1:2,1)).',v(i,3)];
    end

end

