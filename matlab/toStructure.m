function data = toStructure(dataFile)
% Convert data in a file to a structure array
% dataFile is a file name.
% data is a structure array.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Read in dataFile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Nheaderlines = 148;

D = importdata(dataFile, ' ', 148);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert all of data into structure array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialize structure array
[N,col] = size(D.textdata); % number of galaxies + Nheaderlines, fields

field = cell(1,col);

for i = 1:col
    field{i} = D.textdata{Nheaderlines,i};
    data.(field{i}) = [];
end

% Insert data into initialized data structure array
for i=1:N-Nheaderlines
    for j=1:col
        if strcmp(field{j},'curve_used')
            data(i).(field{j}) = D.textdata(i+Nheaderlines,j);
        else
            data(i).(field{j}) = str2double(D.textdata(i+Nheaderlines,j));
        end
    end
    
    %{
    for j=1:col
        data(i).(field{j}) = D.data(i,j);
    end
    %}
end

end