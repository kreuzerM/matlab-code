function [d] = getDistance(p1,p2)
% get distance between points p1(x1,y1,z1) and p2(x2,y2,z2)
    d = sqrt(sum((p2-p1).^2));
end

