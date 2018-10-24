function [theta] = getTrueDOA(p1,p2,rot)
% calculate DOA of p2 towards p1 in x-y plane  
  theta = atan2d(p2(2)-p1(2), p2(1)-p1(1));
  theta = round(theta - rot);
  theta = wrapTo180(theta);
  
end

