%% experiment with rir generator
%params;
c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
n = 1024;                   % Number of samples
%% room dimensions
L = [5 6 2.8];   % Room dimensions [x y z] (m)    
%% microphone positions
r1 = [2.5 3 1.21];              % Receiver position [x y z] (m)
r2 = [2.7 3 1.21];
r3 = [2.9 3 1.21];
r4 = [3.1 3 1.21];
r5 = [3.2 3 1.21];
r6 = [3.4 3 1.21];
r = [r1;r2;r3;r4;r5;r6];

%% source positions
s1 = [4.5 5 1.4]; 
s2 = [2 4 1.4];% Source position [x y z] (m)
s = [s1;s2];

beta = 0.12;                 % Reverberation time (s)
h =10 * wrap_rir_generator(c, fs, r, s, L, beta,'dim',3,'nsample',2048,'mtype','omnidirectional'); 
%h1 = wrap_rir_generator(cfg.c, cfg.fs, cfg.position_mic, cfg.position_speaker, cfg.room_dim, cfg.beta, cfg.L);
