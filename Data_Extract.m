clear all;
close all;
clc;

minSample = 151;
maxSample = 151;
SStr = 'Sample00';
DStr = '_data.mat';

filename = 'Selected.txt';
delimiterIn = ' ';
headerlinesIn = 0;
[usedGestures] = textread(filename,'%s');

if (strcmp(usedGestures(1), 'sonostufo'))
    disp(usedGestures(1));
end

for i = minSample : maxSample
    %disp(i);
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
    
    %disp(strfind(labels(:).Name, 'sonostufo'));
    %disp(length(usedGestures));
    gestureInFile = 0;
    gesturesLoc = [];
    gestureTime = [];
    for j = 1:length(usedGestures)
        for k = 1:length(labels)
            if (strcmp(usedGestures(j),labels(k).Name))
                gestureInFile = gestureInFile + 1;
                gesturesLoc = [gesturesLoc; k];
                gestureTime = [gestureTime; labels(k).End - labels(k).Begin];
            end
        end
    end
    
    %WARNING: Hardoded to work only with 2 used gestures!
    if (gestureInFile == 2)
        if (abs(gesturesLoc(1) - gesturesLoc(2)) == 1)
            connectedGestures = 1;
        else 
            connectedGestures = 0;
        end
    end
    
    %If NN trains junk gestures as gestures, increase junkTime constant!
    junkFramesBegin = 7 + max(round(normrnd(mean(gestureTime-7),mean(gestureTime-7)/2.5, 2-connectedGestures, 1)),0);
    junkFramesEnd = 10 + max(round(normrnd(mean(gestureTime-10),mean(gestureTime-10)/2.5, 2-connectedGestures, 1)),0);
    
    frames = st.Frames;
    
    if (connectedGestures == 1)
        
        textFile = fopen(strcat('Skeletons00', int2str(i) ,'.txt'),'w');
        startingLoc = 1;
        endingLoc = numFrames;
        
        if (lables(min(gestureLoc)).Begin - junkFramesBegin > 1)
            startingLoc = lables(min(gestureLoc)).Begin - junkFramesBegin;
            disp(startingLoc);
        end
        if (labels(max(gestureLoc)).End + junkFramesEnd < numFrames)
            endingLoc = labels(max(gestureLoc)).End + junkFramesEnd;
            disp(endingLoc);
        end
        
        for j = startingLoc : endingLoc
            currFrame = frames(j).Skeleton;
            wpos = currFrame.WorldPosition;
            wrot = currFrame.WorldRotation;
            pix = currFrame.PixelPosition;
            joints = currFrame.JointType;    
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
    end
    
    if (connectedGestures == 0)
        for a = 1 : gestureInFile
            
            textFile = fopen(strcat('Skeletons0', int2str(a), int2str(i) ,'.txt'),'w');
            startingLoc = 0;
            endingLoc = numFrames;
            
            if (labels(gesturesLoc(a)).Begin - junkFramesBegin(a) > 1) 
                startingLoc = labels(gesturesLoc(a)).Begin - junkFramesBegin(a);
                disp(startingLoc);
            end
            if (labels(gesturesLoc(a)).End + junkFramesEnd(a) < numFrames)
                endingLoc = labels(gesturesLoc(a)).End + junkFramesEnd(a);
                disp(endingLoc);
            end
            
            for j = startingLoc : endingLoc
                currFrame = frames(j).Skeleton;
                wpos = currFrame.WorldPosition;
                wrot = currFrame.WorldRotation;
                pix = currFrame.PixelPosition;
                joints = currFrame.JointType;    
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
        end
    end
    
    
    
    textFile = fopen(strcat('Header00', int2str(i), '.txt'),'w');
    fprintf(textFile,'%d\n%d\n%d', numFrames, frameRate, maxDepth);
end


