function [ signal ] = getSourceSignal(path,fs,siglen )
% Lood source file from path, lengthen or shorten the signal to meet the
% specified length and adjust sampling frequency
    [x,sf] = audioread(path);
    %% adjust sampling frequency
    if(sf ~= fs)
        x = resample(x,1,sf/fs);
    end
    %% loop or shorten signal
    if siglen*fs < length(x)
        x = x(1:(siglen*fs),:);
    elseif siglen*fs > length(x)
        diff = (siglen*fs) - length(x);
        k = floor(diff/ length(x));
        r = ceil(mod(diff,length(x)));
        tmp=x;
        for i = 1:k
            x=[x;tmp];
        end
        x = [x;tmp(1:r,:)];
    end

    signal = x;
end

