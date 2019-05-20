function ratioZ_plot_lowess(Vdata,Wdata,Udata,method,BPT,temp)
% Plots metallicity method against the ratio of dark matter halo mass to 
% stellar mass.
% data are structure arrays.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                            FILTER GALAXIES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Vdata = filterGalaxies(Vdata, BPT, temp);
Wdata = filterGalaxies(Wdata, BPT, temp);
Udata = filterGalaxies(Udata, BPT, temp);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                            METHOD PARAMETERS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Zmin,Zmax,label] = abundParams(method);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                           CALCULATE CONSTANTS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
% Number of galaxies in plot
n = sum(~isnan([Vdata.(method)])) + sum(~isnan([Wdata.(method)])) +...
    sum(~isnan([Udata.(method)]));

% Correlation coefficient
r = corr([Vdata.Mdark_Mstar_ratio Wdata.Mdark_Mstar_ratio Udata.Mdark_Mstar_ratio]',...
    [Vdata.(method) Wdata.(method) Udata.(method)]', 'rows','pairwise');
r_err = corrcoefError(r,n);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         LOCAL LINEAR REGRESSION
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Remove galaxies with nan values
v_boolarray = ~isnan([Vdata.(method)]);
w_boolarray = ~isnan([Wdata.(method)]);

Vdata = Vdata(v_boolarray);
Wdata = Wdata(w_boolarray);

% Sort arrays
[Vsort_x, v_index] = sort([Vdata.(method)]);
[Wsort_x, w_index] = sort([Wdata.(method)]);

V_y = log10([Vdata.Mdark_Mstar_ratio]);
W_y = log10([Wdata.Mdark_Mstar_ratio]);


V_smooth = smooth(Vsort_x, V_y(v_index), 0.5, 'rlowess');
W_smooth = smooth(Wsort_x, W_y(w_index), 0.5, 'rlowess');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                               PLOTTING
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% y-axis limits
y_min = 0;
y_max = 2.5;


% formatting
tSize = 18;     % text size
pointSize = 30; % scatter point size
lwidth = 2;     % Line width


figure
hold on
Uplot = scatter([Udata.(method)],log10([Udata.Mdark_Mstar_ratio]), pointSize,'g',...
    'MarkerEdgeAlpha',0.5, 'DisplayName','Uncertain');
Wplot = scatter([Wdata.(method)],log10([Wdata.Mdark_Mstar_ratio]), pointSize,'^','fill','k',...
    'MarkerFaceAlpha',0.5, 'DisplayName','Wall');
Vplot = scatter([Vdata.(method)],log10([Vdata.Mdark_Mstar_ratio]), pointSize,'fill','r',...
    'MarkerFaceAlpha',0.5, 'DisplayName','Void');
plot(Wsort_x, W_smooth, 'k', 'lineWidth',lwidth)
plot(Vsort_x, V_smooth, 'r', 'lineWidth',lwidth)
hold off
axis([Zmin Zmax y_min y_max])
L = legend([Vplot, Wplot, Uplot], 'Location','NorthWest');
xlabel(label, 'fontSize',tSize)
ylabel('log(M_{DM}/M_*)', 'fontSize',tSize)
%{
% correlation coefficient
text((Zmin + 0.25),(y_max - 0.15), sprintf('r = %.2f \\pm %.3f',r,r_err),...
    'fontSize',(tSize-2), 'margin',5)
% number of galaxies
text((Zmax - 0.25),(y_min + 0.15), sprintf('n = %d',n), 'fontSize',(tSize-2),...
    'horizontalAlignment','right', 'fontAngle','italic')
%}
set(gca, 'fontSize',tSize-2, 'lineWidth',4, 'tickLength',2*get(gca,'tickLength'))
L.LineWidth = 1;