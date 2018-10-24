
%% clear
clean;
%% generate source signals
config;
cfg.sig_len = 5;
[tmp,fs] = audioread(cfg.source_path{1});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s1 = tmp;
var1 = var(s1,1);

[tmp,fs] = audioread(cfg.source_path{5});
tmp = resample(tmp,1,fs/cfg.fs);
tmp = tmp(1:cfg.sig_len*cfg.fs);
s2 = tmp;
var2= var(s2,1);
s2 = s2* sqrt(var1/var2);

[tmp3,fs] = audioread(cfg.source_path{5});
tmp3 = resample(tmp3,1,fs/cfg.fs);
tmp3 = tmp3(1:cfg.sig_len*cfg.fs);
s3 = tmp3;
s = [s1,s2,s3];
cfg.n_src = 2;
%% generate microphone signals
H = wrap_rir_generator(1,cfg);
for q = 1:cfg.n_src
    x(:,:,q) = fftfilt(squeeze(H(:,q,:)),s(:,q));
end
 x = sum(x,3);
 scalefac = max(max(abs(x)));
 x = x./scalefac;
 xn = addNoise(x,cfg.SNR);


%% calculate DOA
[Pmu_wide,theta, freq] = wideMusicDOA(xn,cfg.res,cfg);
[Pmu_issm,DOA] = issmMusicDOA(Pmu_wide,theta,cfg);
 %% show Room
 f=visualizeSetup(cfg,1);
%% DOA plot
figure;
set(0,'defaulttextinterpreter','latex')
hold on;
grid on;
grid minor;
plot(theta,10*log10(Pmu_issm));
x_lim = xlim;
y_lim = ylim;
xlabel('$\theta[^\circ]$','FontSize',20);
ylabel('$P_{\mathrm{music}}[dB]$','FontSize',20)
for i=1:cfg.n_src
plot([DOA(i) DOA(i)],[y_lim(1) y_lim(2)], 'r--');
text(DOA(i)+4,y_lim(2)*0.95,['DOA ',num2str(i), ': ',num2str(DOA(i)), '$^\circ$'],'FontSize',8);
end

%% RIRs for second array 
H2 = wrap_rir_generator(2,cfg);
for q = 1:cfg.n_src
    x2(:,:,q) = fftfilt(squeeze(H2(:,q,:)),s(:,q));
end
 x2 = sum(x2,3);
 scalefac = max(max(abs(x2)));
 x2 = x2./scalefac;
 x2n = addNoise(x2,cfg.SNR);

%% second DOAs
[Pmu_wide2,theta2, freq2] = wideMusicDOA(x2n,cfg.res,cfg);
[Pmu_issm2,DOA2] = issmMusicDOA(Pmu_wide2,theta,cfg);
DOA = [ DOA;DOA2];
%% visualize results
%% add DOA lines for intersection to figure f;
figure(f);
set(f,'defaulttextinterpreter','latex')
for k=1:cfg.n_array
        for i = 1:cfg.n_src
            plot([cfg.pos_ref(k,1), cfg.pos_ref(k,1) + cosd((DOA(k,i)+cfg.mic_array_rot(k)))*8], [cfg.pos_ref(k,2), cfg.pos_ref(k,2)+ sind((DOA(k,i)+cfg.mic_array_rot(k)))*8], 'g');
          
           
        end
        chH = get(gca,'Children');
        legend([chH(end-3),chH(end-5),chH(end-6),chH(1)],{'microphone positions','source positions','reference positions','estimated LOS'},'FontSize',10)
end
%% doa plot array 2
figure;
set(0,'defaulttextinterpreter','latex')
hold on;
grid on;
grid minor;
plot(theta,10*log10(Pmu_issm2));
x_lim = xlim;
y_lim = ylim;
xlabel('$\theta[^\circ]$','FontSize',20);
ylabel('$P_{\mathrm{music}}[dB]$','FontSize',20)
for i=1:cfg.n_src
plot([DOA2(i) DOA2(i)],[y_lim(1) y_lim(2)], 'r--');
text(DOA2(i)+4,y_lim(2)*0.95,['DOA ',num2str(i), ': ',num2str(DOA2(i)), '$^\circ$'],'FontSize',8);
end
%%  find Intersections of DOA Lines
    line_length = 8;
    points = 100000;
    % line 1
    
    x1_start = cfg.pos_ref(1,1);
    x1_end = x1_start+ cosd(DOA(1,1)+cfg.mic_array_rot(1))*line_length;
    y1_start = cfg.pos_ref(1,2);
    y1_end = y1_start + sind(DOA(1,1)+cfg.mic_array_rot(1))*line_length;
    g1 = [linspace(x1_start,x1_end,points);linspace(y1_start,y1_end,points)];

    
    % line 2
    x2_start = cfg.pos_ref(2,1);
    x2_end = x1_start+ cosd(DOA(2,1)+cfg.mic_array_rot(2))*line_length;
    y2_start = cfg.pos_ref(2,2);
    y2_end = y1_start + sind(DOA(2,1)+cfg.mic_array_rot(2))*line_length;
    g2 = [linspace(x2_start,x2_end,points);linspace(y2_start,y2_end,points)];
  
    
    % line 3
    x3_start = cfg.pos_ref(1,1);
    x3_end = x3_start+ cosd(DOA(1,2)+cfg.mic_array_rot(1))*line_length;
    y3_start = cfg.pos_ref(1,2);
    y3_end = y1_start + sind(DOA(1,2)+cfg.mic_array_rot(1))*line_length;
    g3 = [linspace(x3_start,x3_end,points);linspace(y3_start,y3_end,points)];
  
    
     % line 2
    x4_start = cfg.pos_ref(2,1);
    x4_end = x4_start+ cosd(DOA(2,2)+cfg.mic_array_rot(2))*line_length;
    y4_start = cfg.pos_ref(2,2);
    y4_end = y4_start + sind(DOA(2,2)+cfg.mic_array_rot(2))*line_length;
    g4 = [linspace(x4_start,x4_end,points);linspace(y4_start,y4_end,points)];
    
    figure;hold on;
    plot(g1(1,:),g1(2,:));
    plot(g2(1,:),g2(2,:));
    plot(g3(1,:),g3(2,:));
    plot(g4(1,:),g4(2,:));
    scatter(cfg.pos_ref(:,1),cfg.pos_ref(:,2),'+');
    scatter(cfg.source_pos(1:cfg.n_src,1),cfg.source_pos(1:cfg.n_src,2),'or');
    legend({'g1','g2','g3','g4','ref pos', 'source pos'});
    
    th = 0.3;
    % intersections between g1 and g2
    
    diff1 =  sqrt((sum((g1-g2).^2)));
     %ix1 = find(diff1 < th);
    [~,ix1] = min(diff1);
    p1 = g1(:,ix1);
    p11 = g2(:,ix1);
    scatter([p1(1) p11(1)], [p1(2) p11(2)],'xm');
    
    % intersections between g3 and g4
    
    diff2 = sqrt((sum((g3-g4).^2)));
    [~,ix2] = min(diff2);
    %ix2 = find(diff2 < th);
    p2 = g3(:,ix2);
    p22 = g4(:,ix2);
    scatter([p2(1) p22(1)],[p2(2) p22(2)],'xc');
     legend({'g1','g2','g3','g4','ref pos', 'source pos','p1','p2'});
%%
% 
% 
% g1 = [x1;y1];
% g2 = [x2;y2];
% [idx] =find(abs(g1-g2) < 0.0001);
% x = x1(idx(1));
% y = y1(idx(1));


