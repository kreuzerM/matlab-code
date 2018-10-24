function [sensor_pos] = generateSensorArray(ref,nmic,dmic,orientation)
% generate coordinates for sensors in an ULA
%   
    if mod(nmic,2) ~= 0
     error('sensor array not symmetric'); 
    end
    % ---- o - o 
    n = [nmic/2:-1:1,-1:-1:-nmic/2].';
    sensor_pos(:,2) = (ref(2) + sign(n).*dmic/2 +1*sign(n).*(dmic .*(abs(n)-1))); 
    
    sensor_pos(:,1) =  ref(1)*ones(nmic,1);
    sensor_pos(:,3) = ref(3)*ones(nmic,1);
    
    % orientation
   v = sensor_pos-ref;
   vR = rotate(v,orientation);
   sensor_pos= vR +ref;
    
end

