function scatter_plot_rlowess(Vdata,Wdata,Udata,fieldx,fieldy)
% Plots galaxy property (fieldy) as a function of galaxy property (fieldx)
% _data are structure arrays


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter galaxies
%--------------------------------------------------------------------------
vbool_array = [Vdata.curve_used] ~= -99;
Vdata = Vdata(vbool_array);

wbool_array = [Wdata.curve_used] ~= -99;
Wdata = Wdata(wbool_array);

ebool_array = [Udata.curve_used] ~= -99;
Udata = Udata(ebool_array);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method parameters
%--------------------------------------------------------------------------
[xmin, xmax, x_label] = params(fieldx);
[ymin, ymax, y_label] = params(fieldy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate constants
%--------------------------------------------------------------------------
% Number of galaxies in the plot
n = sum(isfinite([Vdata.(fieldx)] + [Vdata.(fieldy)])) +...
    sum(isfinite([Wdata.(fieldx)] + [Wdata.(fieldy)])) +...
    sum(isfinite([Udata.(fieldx)] + [Udata.(fieldy)]));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local linear regression
%--------------------------------------------------------------------------
% Remove galaxies with nan values
v_boolarray = isfinite([Vdata.(fieldx)]) & isfinite([Vdata.(fieldy)]);
w_boolarray = isfinite([Wdata.(fieldx)]) & isfinite([Wdata.(fieldy)]);

Vdata = Vdata(v_boolarray);
Wdata = Wdata(w_boolarray);

% Sort arrays
[Vsort_x, v_index] = sort([Vdata.(fieldx)]);
[Wsort_x, w_index] = sort([Wdata.(fieldx)]);

V_y = [Vdata.(fieldy)];
W_y = [Wdata.(fieldy)];


V_smooth = smooth(Vsort_x, V_y(v_index), 0.25, 'rlowess');
W_smooth = smooth(Wsort_x, W_y(w_index), 0.25, 'rlowess');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%--------------------------------------------------------------------------
% Plot formatting
tSize = 18;     % text size
pointSize = 30; % scatter point size
lwidth = 2;     % Line width


figure
hold on

Uplot = scatter([Udata.(fieldx)],[Udata.(fieldy)], pointSize,'g',...
    'MarkerEdgeAlpha',0.2, 'DisplayName','Uncertain');
Wplot = scatter([Wdata.(fieldx)],[Wdata.(fieldy)], pointSize,'^','fill','k',...
    'MarkerFaceAlpha',0.2, 'DisplayName','Wall');
Vplot = scatter([Vdata.(fieldx)],[Vdata.(fieldy)], pointSize,'fill','r',...
    'MarkerFaceAlpha',0.2, 'DisplayName','Void');

plot(Wsort_x, W_smooth, 'k', 'lineWidth',lwidth);
plot(Vsort_x, V_smooth, 'r', 'lineWidth',lwidth);

hold off

%axis([xmin xmax+0.05 ymin ymax])

L = legend([Vplot, Wplot, Uplot]);
L.Location = 'SouthEast';

xlabel(x_label, 'fontSize',tSize)
ylabel(y_label, 'fontSize',tSize)

% number of galaxies
text((xmin + 0.05),(ymax - 0.1),sprintf('n = %d',n), 'fontSize',(tSize-2),...
    'fontAngle','italic')

set(gca, 'fontSize',tSize-2, 'lineWidth',4, ...
    'tickLength',2*get(gca,'tickLength'), 'YScale','log')
L.LineWidth = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%