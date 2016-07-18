clear all;
close all;
clc;

minSample = 151;
maxSample = 151;
SStr = 'Sample00';
DStr = '_data.mat';

for i = minSample : maxSample
    inString = strcat(SStr, int2str(i), DStr);
    st = load(inString);
    st = st.Video;
    numFrames = st.NumFrames;
    frameRate = st.FrameRate;
    maxDepth = st.MaxDepth;
    labels = st.Labels;
    labNames = labels(:).Name;
    textFile = fopen(strcat('Labels00', int2str(i) ,'.txt'),'w');
    for j = 1:length(labels)
       fprintf(textFile, '%s %d %d \n', labels(j).Name, labels(j).Begin, labels(j).End);
    end
    
    frames = st.Frames;
    textFile = fopen(strcat('Skeletons00', int2str(i) ,'.txt'),'w');
    for j = 1:length(frames)
        currFrame = frames(j).Skeleton;
        wpos = currFrame.WorldPosition;
        wrot = currFrame.WorldRotation;
        pix = currFrame.PixelPosition;
        joints = currFrame.JointType;
        %T = cell2table(joints,'VariableNames',{'Name'});
        %csvwrite('blabla.csv',joints);
        %A = cell2mat(joints);
        for k = 1:length(joints)
            %fprintf(textFile, '%s ', joints(k){:});
            for l = 1:size(wpos,2)
                fprintf(textFile, '%d ', wpos(k,l));
            end
            for l = 1:size(wrot,2)
                fprintf(textFile, '%d ', wrot(k,l));
            end
            for l = 1:size(pix,2)
                fprintf(textFile, '%d ', pix(k,l));
            end
            fprintf(textFile, '\n');
        end
    end
    
    textFile = fopen(strcat('Header00', int2str(i), '.txt'),'w');
    fprintf(textFile,'%d\n%d\n%d', numFrames, frameRate, maxDepth);
end


