function Pos = generateSourcePosRandom(nsrc,width,length,height,origin)
%generate random source positions for a specified enclosure
%   nsrc: number of positions to be generated
%   width: width of the enclosure in which the positions are to be generated [x]
%   length: length of the enclosure in which the postions are to be generated [y]
%   height: z coordinate of the generated source positions
        % annotation: for the moment fixed z coordinates, could be extended
        % to random positions as well
xrand = width*rand(nsrc,1);
yrand = length*rand(nsrc,1);
Pos = [xrand+origin(1),yrand+origin(2), height*ones(nsrc,1)];
end

