function h = wrap_rir_generator(array_index,cfg, varargin)
%% wrap function for rir_generator
% * *input:*
% * *cfg.c:* sound velocity in m/s
% * *cfg.fs:* samplinc frequency in Hz
% * *cfg.pos_sensor:* N x 3 matrix containing sensor positions [x y z]
% * *cfg.pos_source:* M x 3 matrix containing speaker positions [x y z]
% * *cfg.room*: 1 x 3 vector containing room dimensions [x y z]
% * *cfg.beta*: 1 x 6 vector specifying the reflection coefficients or
%    reverberation time  in s
%% optional input arguments
% * can be modified using key-value pairs:
% * function(param1,param2,'key1',value1, 'key2',value2)
% * *nsample*: number of samples to calculate, default: n = beta*fs
% * *mtype:*: string describing the type of microphone that is used. possible values:
%             ['omnidirectional','subcardioid', 'cardioid', 'hypercardioid','bidirectional']
%              default: 'omnidirectional
% * *order*: maximum reflection order, default: -1
% * *dim*: room dimension, possible values: 2,3
% * *orientation:* direction in which the microphone is pointed, specified
%                  using a 1x2 vector: [azimuth elevation] in radions
% * *hpfilter*: use 'false' or 0 to disable high pass filter
% usage example: h = wrap_rir_generator(c, fs, r, s, d, beta,'dim',2,'mtype','bidirectional','orientation',[ pi 0])
% * *output:*
% * *h*: n x N x M matrix containing the Room Impulse Response(RIR) 
% samples x source x mic

%% default parameters
    n = cfg.beta *cfg.fs;
    m_type = 'omnidirectional';
    possible_m_types = {'omnidirectional','subcardioid','hypercardioid','bidirectional'};
    order = -1;
    dim = 3;
    orientation = [0 0];
    hp_filter = 1;
    %% 
    if (nargin < 1)
    error('wrong number of input arguments')
    end
    %% input validation functions
        function result = checkdim(x)
        switch x
            case 2
                result = true;
            case 3
                result = true;
            otherwise
                error('False input! possible values for dim:2,3')
                result = false;
        end           
        end
        function result = checkorientation(x)
        if isequal(size(x),[1 2]) 
            result = true;
        else
            result = false;
            error('False input! input format for orientation: 1x2 vector containing [azimuth elevation] in radians');
        end
        end
    %% input parser
    p = inputParser;
    p.KeepUnmatched = true;
    addOptional(p,'nsample',n,@(x) isnumeric(x) && (x > 0));
    addOptional(p,'mtype',m_type,@(x) any(validatestring(x,possible_m_types)));
    addOptional(p,'order',order,@(x) isnumeric(x) && (x>0));
    addOptional(p,'dim',dim,@checkdim);
    addOptional(p,'orientation',orientation,@checkorientation);
    addOptional(p,'hpfilter',hp_filter,@islogical);
    parse(p,varargin{:});
    %%
    mic_pos = cfg.mic_pos(:,:,array_index);
    for i=1:cfg.n_src
        tmp =  rir_generator(cfg.c, cfg.fs, mic_pos, cfg.source_pos(i,:), cfg.room_dim,cfg.beta,p.Results.nsample,p.Results.mtype,p.Results.order,p.Results.dim,p.Results.orientation,p.Results.hpfilter);
        h(:,i,:) = tmp';
    end
    end