% This function loads all .csv files in a user-selected directory into a DATA array 
% The files should have frequency in the second column and a header row, as
% is typical for Rocca output of whistle contours
function ARTwarp_Load_JSON_Data

pkg load json
global DATA numSamples tempres

path = '/home/loic/Data/DOLPHINFREE/Scripts/SIGWI/resources/DF-whistles/smooth_per_day/500ms/20200711'; 
path = [path '/*json'];
DATA = dir(path);
DATA = rmfield(DATA,'date');
DATA = rmfield(DATA,'datenum');
DATA = rmfield(DATA,'bytes');
DATA = rmfield(DATA,'isdir');
[numSamples x] = size(DATA);


for c1 = 1:numSamples
    
    fid = fopen(fullfile(DATA(c1).folder, DATA(c1).name)); 
    raw = fread(fid,inf); 
    str = char(transpose(raw)); 
    fclose(fid); 
 
    % Parse JSON using Java
    jsonData = jsondecode(str);

    DATA(c1).ctrlength = jsonData.time(length(jsonData.time))-jsonData.time(1);
    DATA(c1).length = length(jsonData.time);
    DATA(c1).contour = transpose(jsonData.frequency);
    DATA(c1).tempres = DATA(c1).ctrlength/DATA(c1).length;
    DATA(c1).category = 0;


end

h = findobj('Tag', 'Runmenu');                                                                                                                        
set(h, 'Enable', 'on');
disp(path);