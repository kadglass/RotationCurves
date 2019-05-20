function [mininum, maximum, label] = abundParams(abundance)
% Define plot parameters (axis minimum, maximum, and label) for a given 
% abundance.


% N/H
if strcmp(abundance,'N12logNH')
    mininum = 5.5;
    maximum = 8.5;
    label = '12 + log(N/H)';

% N+/H
elseif strcmp(abundance,'N12logNpH')
    mininum = 5.5;
    maximum = 8.5;
    label = '12 + log(N^+/H)';

% N/O
elseif strcmp(abundance,'logNO')
    mininum = -2;
    maximum = -0.5;
    label = 'log(N/O)';

% N+/O+
elseif strcmp(abundance,'logNpOp')
    mininum = -2;
    maximum = 0.5;
    label = 'log(N^+/O^+)';

% O/H
elseif strcmp(abundance,'Z12logOH') || strcmp(abundance,'Z12logOH_min')
    mininum = 7;
    maximum = 8.5;
    label = '12 + log(O/H)';
    
elseif strcmp(abundance,'T04_data')
    mininum = 6.5;
    maximum = 9.5;
    label = '12 + log(O/H) (Tremonti et al. 2004)';

% O+/H
elseif strcmp(abundance,'Z12logOpH')
    mininum = 6.5;
    maximum = 9;
    label = '12 + log(O^+/H)';
    
% total SFR
elseif strcmp(method,'totSFR')
    label = 'log(SFR) [M_{sol}/yr]';
    mininum = -2;
    maximum = 2;

% Not defined
else
    print 'WARNING: abundance not known.'
    return
    
end