function data = filterGalaxies(data, BPT, temp, Mr_min,Mr_max, temp_flag, flag3727, flagS2, flagT04, flag4363AN)
% Remove galaxies based on different requirements


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                            DEFAULT VALUES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~exist('Mr_min','var')
    Mr_min = 0;
end

if ~exist('Mr_max','var')
    Mr_max = -30;
end

if ~exist('BPT','var')
    BPT = -1;
end

if ~exist('temp','var')
    temp = 3;
end

if ~exist('temp_flag','var')
    temp_flag = 1;
end

if ~exist('flag3727','var')
    flag3727 = 0;
end

if ~exist('flagS2','var')
    flagS2 = 0;
end

if ~exist('flagT04','var')
    flagT04 = 0;
end

if ~exist('flag4363AN','var')
    flag4363AN = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                            FILTER GALAXIES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Remove galaxies with mass ratios of -1 (not determined)
bool_array = [data.Mdark_Mstar_ratio] == -1;
data = data(~bool_array);


% Remove galaxies outside absolute magnitude range
bool_array = [data.rabsmag] <= Mr_min & [data.rabsmag] >= Mr_max;
data = data(bool_array);
% (The relations appear to be opposite what one would expect because
% absolute magnitudes are all negative.)


% Only use those galaxies with temperatures t3 < temp
if temp_flag
    if ismember('t3',fieldnames(data))
        bool_array = [data.t3] < temp;
        data = data(bool_array);
    elseif ismember('T3',fieldnames(data))
        bool_array = [data.T3] < temp*10000;
        data = data(bool_array);
    end
end


% Retain only those galaxies with specified BPT classification
if BPT ~= -1
    bool_array = [data.BPT] == BPT;
    data = data(bool_array);
end


% Separate galaxies based on reliability of [OII] 3727
if flag3727
    bool_array = [data.flag3727] == 1;
    data = data(bool_array);
end


% Only use those galaxies with [SII] 6731/6717 > 0.68
if flagS2
    bool_array = [data.flagS2] == 1;
    data = data(bool_array);
end


% Only use those galaxies with T04 data
if flagT04
    bool_array = ~isnan([data.T04_data]);
    data = data(bool_array);
end


% Remove all galaxies with [OIII] 4363 A/N < 1
if flag4363AN
    bool_array = [data.OIII_4363_AoN] >= 1;
    data = data(bool_array);
end