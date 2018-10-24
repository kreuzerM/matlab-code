%% Configuration file for Simulation Framework 
    %% specification of all important parameters

    %% parameters concering room geometry
% room dimensions in [x y z] in m
cfg.room_dim = [8 10 3.5];


% beta: either reverberation time(t60) of the specified room in s or or
%       reflection coefficients for each wall
cfg.beta = 0.4;
%speed of sound in m/s
cfg.c = 342;
% number of microphones:
cfg.n_mic = 4;
% number of microphone arrays
cfg.n_array = 2;
% distance between microphones in m
cfg.d_mic = 0.05;
% rotation of the microphone arrays
cfg.mic_array_rot = [30,135];
% positions of the microphone array centers
cfg.pos_ref = [2,1,1.5; 4.5, 1, 1.5];
cfg.mic_pos = zeros(cfg.n_mic,3,cfg.n_array);
for i=1:cfg.n_array
    cfg.mic_pos(:,:,i) = generateSensorArray(cfg.pos_ref(i,:),cfg.n_mic,cfg.d_mic,cfg.mic_array_rot(i));
end



%% source signals
% sampling frequency in Hz
cfg.fs = 16000;
% number of sources
cfg.n_src = 2;
% path(s) of the source file(s)
cfg.source_path = {'source_signals/Male1.wav','./source_signals/Male2.wav', './source_signals/Male3.wav',  './source_signals/Female1.wav', ...
    './source_signals/Female3.wav', './source_signals/Female4.wav','./source_signals/vacuum_cleaner.wav','./source_signals/water_pouring.wav',...
    './source_signals/hair_dryer.wav','./source_signals/coffee_machine.wav','./source_signals/keyboard.wav','./source_signals/water_stir.wav',...
    './source_signals/coughing.wav','./source_signals/snoring.wav','./source_signals/telephone_ring.wav'};
% signal length in s 
cfg.sig_len = 5;
% SNR in dB
cfg.SNR = 25;
% source positions
cfg.source_pos(1,:) = generateSourcePosRef(cfg.pos_ref(2,:), 2 ,10,cfg.mic_array_rot(2));
cfg.source_pos(2,:) = generateSourcePosRef(cfg.pos_ref(1,:), 1,-20,cfg.mic_array_rot(1));
cfg.source_pos(3,:) = generateSourcePosRef(cfg.pos_ref(1,:), 0.5 ,0,cfg.mic_array_rot(1));
%% STFT parameters for CDR estimation
% window length in s
cfg.cdr.t_window = 0.025;
% window length in samples
cfg.cdr.n_window = round(cfg.cdr.t_window*cfg.fs);          
% frame shift in s
cfg.cdr.t_fshift = 0.010;
% frame shift in samples
cfg.cdr.n_fshift= round(cfg.cdr.t_fshift*cfg.fs); 
% overlap in samples
cfg.cdr.n_overlap = cfg.cdr.n_window - cfg.cdr.n_fshift;        
% window
cfg.cdr.window = hann(cfg.cdr.n_window); 
% nummber fft bins for STFT
cfg.cdr.n_fft = 2^(ceil(log(cfg.cdr.n_window)/log(2)));
%frequency vector
cfg.cdr.freq = ((0:cfg.cdr.n_fft/2)*(cfg.fs/cfg.cdr.n_fft)).';           
% frequency range 125 Hz - 3500 Hz fbin_index= f*nfft/fs +1
%cfg.freq_range = [125, 3500];
%cfg.cdr.freq_range = [300,1000]; 
cfg.cdr.freq_range = [200,4000]; 

% frequency bin indices corresponding to cfg.cdr.freq_range
cfg.cdr.freq_range_bins = [round(cfg.cdr.freq_range(1) * cfg.cdr.n_fft/cfg.fs)+1,round(cfg.cdr.freq_range(2) * cfg.cdr.n_fft/cfg.fs)+1];
%cfg.freq_range = [5:113];
% smoothing factor for psd estimation
cfg.cdr.lambda = 0.95;

%% STFT parameters for MUSIC DOA
%angular resolution
cfg.res=1; 
% window length in s
cfg.music.t_window = 0.025 %0.05;
%frame shift in s
cfg.music.t_fshift = 0.010 %0.05;  
% window length in samples
cfg.music.n_window = round(cfg.music.t_window*cfg.fs);  
%frame shift in samples
cfg.music.n_fshift = round(cfg.music.t_fshift*cfg.fs); 
% frame overlap in samples
cfg.music.n_overlap = cfg.music.n_window - cfg.music.n_fshift;
% window
cfg.music.window = hann(cfg.music.n_window);%rectwin(cfg.music.n_window); 
% fft length
cfg.music.n_fft = 2^(ceil(log(cfg.music.n_window)/log(2)));
% frequency column vector
cfg.music.freq = ((0:cfg.music.n_fft/2)*(cfg.fs/cfg.music.n_fft)).';
% evaluated frequency range [fmin,fmax] [Hz]
cfg.music.freq_range = [200,4000]; 
%cfg.music.freq_range = [300,1000];
% indices of the corresponding frequency bins
cfg.music.freq_range_bins = [round(cfg.music.freq_range(1) * cfg.music.n_fft/cfg.fs)+1,...
    round(cfg.music.freq_range(2) * cfg.music.n_fft/cfg.fs)+1];
%% power threshold for STFT bin power
cfg.p_threshold = 0;
