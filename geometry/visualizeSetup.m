function f = visualizeSetup(cfg,anot)

    % visualize Room Setup with sensor and source positions
    % plot room

    f=figure('Name', 'Setup','NumberTitle','off');
    set(0,'defaultTextInterpreter','latex');
    set(0,'defaultAxesFontSize',18);
    grid on;
    grid minor;
    hold on;
    pos = [0 0 cfg.room_dim(1) cfg.room_dim(2)];
    room = rectangle('Position',pos,'LineWidth',2.5,'EdgeColor','r');
    markersize = 40;
    axis equal;
    axis([0 cfg.room_dim(1) 0 cfg.room_dim(2)]);
    annotation('textbox',[0.4 0.6 0.2 0.3],'String',['room height: ' ,num2str(cfg.room_dim(3)), ' m'],'FitBoxToText','on','FontSize',10);
    % plot sensors
    for k = 1:cfg.n_array
        h1=scatter(cfg.mic_pos(1:cfg.n_mic,1,k),cfg.mic_pos(1:cfg.n_mic,2,k),markersize,'kx');
        orientation=plot([cfg.pos_ref(k,1) cfg.pos_ref(k,1)+cosd(cfg.mic_array_rot(k))*2], [ cfg.pos_ref(k,2) cfg.pos_ref(k,2)+sind(cfg.mic_array_rot(k))*2], 'k--');
    end
    h2=scatter(cfg.source_pos(1:cfg.n_src,1),cfg.source_pos(1:cfg.n_src,2),markersize,'r*');
    %h3 = scatter(cfg.pos_ref(1:cfg.n_array,1),cfg.pos_ref(1:cfg.n_array,2),markersize,'+k');
    if(anot==1)
        for k = 1:cfg.n_array
            text(cfg.pos_ref(k,1),cfg.pos_ref(k,2)-0.25,num2str(k) ,'FontSize',10,'Color','r');
            for i = 1:cfg.n_mic
                text(cfg.mic_pos(i,1,k),cfg.mic_pos(i,2,k)-0.05,num2str(i) ,'FontSize',8);
            end
        end
       for k = 1:cfg.n_array
            for i = 1:cfg.n_src
                
                
%                 doa = round(getTrueDOA(cfg.pos_ref(k,:),cfg.source_pos(i,:),cfg.mic_array_rot(k)),2);
%                 d = round(getDistance(cfg.pos_ref(k,:),cfg.source_pos(i,:)),2);
%                 text(cfg.source_pos(i,1),cfg.source_pos(i,2)-0.1*k,['Ref ',num2str(k),' $\rho$: ',num2str(d),'m, $\theta$: ',num2str(doa),'$^\circ$'] ,'FontSize',10);
%                 text(cfg.source_pos(i,1),cfg.source_pos(i,2)-0.1*k,[num2str(i)] ,'FontSize',10);
            end
       end
            for i = 1:cfg.n_src
                
                 text(cfg.source_pos(i,1),cfg.source_pos(i,2),num2str(i) ,'FontSize',14);
            end
    end
    legend([h1 h2],{'microphone positions','source positions'},'FontSize',18);
    %legend([h1 h2 h3 ],{'microphone positions','source positions','reference positions'},'FontSize',10);
    xlabel('x[m]');
    ylabel('y[m]');

end

