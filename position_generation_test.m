close all;
clear;
clc;
config;
%% random positions in specified area
cfg.n_src = 7;
cfg.n_array =1;
ref = [4 1.5 1.5];
roi_width = 3;
roi_height = 3;
cfg.source_pos = generateSourcePosRandom(cfg.n_src ,roi_width,roi_height ,1.5,ref);
% generateSourcePosRandom(nsrc,width,length,height,origin)
f = visualizeSetup(cfg,1);
figure(f);
square= rectangle('Position',[ref(1) ref(2) roi_width roi_height] ,'LineWidth',1,'EdgeColor','b');

%% positions with given distance and doa to reference point
cfg.pos_ref = [ 1 7 1.5];
cfg.n_array = 1;
cfg.n_mic = 4;
cfg.mic_array_rot = -45;
cfg.mic_pos = generateSensorArray(cfg.pos_ref,cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(1));

rho = [ 1 2 3].';
theta = [ 10 20 30].';

cfg.source_pos = generateSourcePosRef(cfg.pos_ref,rho ,theta,cfg.mic_array_rot(1));
cfg.n_src = length(cfg.source_pos(:,1))
visualizeSetup(cfg,1);