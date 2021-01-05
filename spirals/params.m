function [minimum, maximum, label] = params(field)
% Define plot parameters (axis minimum, maximum, and label) for a given 
% field name.


% absolute magnitude
if strcmp(field,'rabsmag')
    minimum = -23;
    maximum = -17;
    label = 'M_r';
    
% Ratio of dark matter halo mass to stellar mass
elseif strcmp(field, 'Mdark_Mstar_ratio')
    minimum = 0;
    maximum = 5;
    label = 'M_{DM}/M_*';
end