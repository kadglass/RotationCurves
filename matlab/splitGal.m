function [void, wall, other] = splitGal(all)
% Separates void and wall galaxies
% all is a structure array.
% void, wall, and other are structure arrays


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                           SEPARATE GALAXIES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


vbool_array = [all.vflag] == 1;
wbool_array = [all.vflag] == 0;
obool_array = [all.vflag] == 2;

void = all(vbool_array);
wall = all(wbool_array);
other = all(obool_array);