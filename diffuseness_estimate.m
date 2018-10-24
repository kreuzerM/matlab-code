%clean;
config;

%s_test = zeros(cfg.fs*cfg.sig_len,cfg.n_src);
%% generate source positions
d =  0.15:0.2: 7.5;
cfg.n_array =1;
cfg.pos_ref = [ 1 1 1.5];
cfg.n_src = length(d);
cfg.n_mic = 2;
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
cfg.mic_pos(:,:,1)=generateSensorArray(cfg.pos_ref(1,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(1));
for i=1:cfg.n_src
    cfg.source_pos(i,:) = generateSourcePosRef(cfg.pos_ref(1,:), d(i) ,0,cfg.mic_array_rot(1));
end



%%
[tmp,fs] = audioread(cfg.source_path{1});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s_test = tmp;
%s_test(:,i) = tmp;

%% compute diffuseness
mean_Diff = zeros(cfg.n_src,1);
for i=1:cfg.n_src 
    mean_Diff(i) = compute_feature(1,cfg,s_test,cfg.source_pos(i,:));
end
mean_CDR = 1./mean_Diff -1;
visualizeSetup(cfg,1);
%d = pdist2(cfg.source_pos,[ 1 1 1.5]);
figure('Name','Diffusenes vs Distance');
plot(d,mean_Diff,'-o');
xlabel('Distance[m]');
ylabel('Diffuseness');
figure('Name','CDR vs Distance');
plot(d,mean_CDR,'-o');
xlabel('Distance[m]');
ylabel('CDR');