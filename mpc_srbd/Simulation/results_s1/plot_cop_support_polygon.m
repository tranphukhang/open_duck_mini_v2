clc;
clear;
close all;

% ============================================================
% CoP TRAJECTORY AND SUPPORT POLYGON - STANDING S1 PUSH TEST
%
% Output: 4 separate figures, each 8 x 8 cm
% 1) Full CoP trajectory (0-10 s)
% 2) CoP trajectory near +Y push (2.9-4.0 s)
% 3) CoP trajectory near +X push (5.9-7.0 s)
% 4) Instantaneous support margin versus time
%
% IMPORTANT:
% The polygon drawn in Figures 1-3 is ONE SNAPSHOT,
% NOT a fixed polygon that applies throughout the time window.
% The margin in Figure 4 is logged at each time step and refers
% to the instantaneous active support polygon.
% ============================================================

% ------------------------------------------------------------
% FIND INPUT CSV
% ------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
csv_candidates = dir(fullfile(script_dir, '*timeseries*.csv'));

if numel(csv_candidates) == 1
    csv_file = fullfile(csv_candidates(1).folder, ...
        csv_candidates(1).name);
else
    [csv_name, csv_folder] = uigetfile( ...
        {'*.csv', 'CSV files (*.csv)'}, ...
        'Select standing S1 timeseries CSV', script_dir);
    if isequal(csv_name, 0)
        error('No timeseries CSV file was selected.');
    end
    csv_file = fullfile(csv_folder, csv_name);
end

% ------------------------------------------------------------
% READ CSV AND CHECK REQUIRED COLUMNS
% ------------------------------------------------------------
data = readtable(csv_file);
col_names = data.Properties.VariableNames;

required_columns = { ...
    'time_s', 'actual_cop_total_x_m', 'actual_cop_total_y_m', ...
    'actual_cop_support_margin_m', 'support_active_vertex_count', ...
    'support_mode'};

for foot = {'L', 'R'}
    side = foot{1};
    for corner = 1:4
        required_columns{end+1} = sprintf('foot_%s_corner%d_x_m', side, corner); %#ok<SAGROW>
        required_columns{end+1} = sprintf('foot_%s_corner%d_y_m', side, corner); %#ok<SAGROW>
    end
end

missing_columns = setdiff(required_columns, col_names);
if ~isempty(missing_columns)
    error('Missing CSV columns: %s', strjoin(missing_columns, ', '));
end

t = data.time_s;
cop_x = data.actual_cop_total_x_m;
cop_y = data.actual_cop_total_y_m;
margin = data.actual_cop_support_margin_m;

% ------------------------------------------------------------
% FIGURE STYLE
% ------------------------------------------------------------
fig_width = 8;                  % centimeters
fig_height = 8;                 % centimeters
font_name = 'Times New Roman';
font_axis = 10;
font_label = 11;
font_legend = 7.5;

cop_color = [1, 0, 0];          % red
sole_color = [0, 0.447, 0.741]; % blue
poly_color = [0, 0, 0];         % black

plot_dir = fullfile(script_dir, 'CoP_support_polygon_plots');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ------------------------------------------------------------
% DEFINE 3 TOP-VIEW PLOTS
% Reference times are chosen within valid DSP intervals.
% ------------------------------------------------------------
windows = [ ...
    t(1),  t(end); ...            % Entire test
    2.90, 4.00; ...              % +Y push and recovery
    5.90, 7.00  ...              % +X push and recovery
];
reference_times = [2.90, 3.05, 6.05];
output_names = { ...
    'CoP_trajectory_full', ...
    'CoP_trajectory_push_Y', ...
    'CoP_trajectory_push_X'};

% ------------------------------------------------------------
% DRAW CoP ON SUPPORT POLYGON (3 FIGURES)
% ------------------------------------------------------------
for fig_idx = 1:3

    % Select an instantaneous polygon snapshot in DSP mode.
    eligible = find( ...
        data.support_active_vertex_count >= 3 & ...
        strcmp(string(data.support_mode), 'DSP'));
    if isempty(eligible)
        error('No valid DSP support polygon found in CSV.');
    end
    [~, nearest] = min(abs(t(eligible) - reference_times(fig_idx)));
    kref = eligible(nearest);

    % Active support polygon vertices at this reference time.
    nv = round(data.support_active_vertex_count(kref));
    poly_x = zeros(nv, 1);
    poly_y = zeros(nv, 1);
    for v = 1:nv
        key_x = sprintf('support_active_v%d_x_m', v);
        key_y = sprintf('support_active_v%d_y_m', v);
        if ~ismember(key_x, col_names) || ~ismember(key_y, col_names)
            error('Missing active support polygon vertex columns.');
        end
        values_x = data.(key_x);
        values_y = data.(key_y);
        poly_x(v) = values_x(kref);
        poly_y(v) = values_y(kref);
    end

    % Foot rectangles at the same reference time.
    sole_x = zeros(4, 2);
    sole_y = zeros(4, 2);
    feet = {'L', 'R'};
    for f = 1:2
        for v = 1:4
            col_x = sprintf('foot_%s_corner%d_x_m', feet{f}, v);
            col_y = sprintf('foot_%s_corner%d_y_m', feet{f}, v);
            values_x = data.(col_x);
            values_y = data.(col_y);
            sole_x(v,f) = values_x(kref);
            sole_y(v,f) = values_y(kref);
        end
    end

    % CoP samples from the requested time window.
    sample_mask = t >= windows(fig_idx,1) & ...
                  t <= windows(fig_idx,2) & ...
                  isfinite(cop_x) & isfinite(cop_y);
    indices = find(sample_mask);
    if isempty(indices)
        warning('No CoP values in window %d.', fig_idx);
        continue;
    end

    xc = cop_x(indices);
    yc = cop_y(indices);

    fig = figure('Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5, 5, fig_width, fig_height]);
    set(fig, 'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width, fig_height], ...
        'PaperPosition', [0, 0, fig_width, fig_height], ...
        'PaperPositionMode', 'manual');

    hold on;

    % Support polygon: active-load-based hull at ONE time instant.
    h_poly = plot([poly_x; poly_x(1)], [poly_y; poly_y(1)], ...
        '-', 'Color', poly_color, 'LineWidth', 1.35);

    % Left foot (blue solid), right foot (blue dashed).
    h_left = plot([sole_x(:,1); sole_x(1,1)], ...
        [sole_y(:,1); sole_y(1,1)], '-', ...
        'Color', sole_color, 'LineWidth', 1.0);
    h_right = plot([sole_x(:,2); sole_x(1,2)], ...
        [sole_y(:,2); sole_y(1,2)], '--', ...
        'Color', sole_color, 'LineWidth', 1.0);

    % Actual measured CoP trajectory.
    h_cop = plot(xc, yc, '-', 'Color', cop_color, ...
        'LineWidth', 1.4);

    % Mark start and end of the time window.
    h_start = plot(xc(1), yc(1), 'o', ...
        'Color', cop_color, 'MarkerFaceColor', 'w', ...
        'MarkerSize', 5.5, 'LineWidth', 1.1);
    h_end = plot(xc(end), yc(end), 's', ...
        'Color', cop_color, 'MarkerFaceColor', cop_color, ...
        'MarkerSize', 5.0, 'LineWidth', 1.1);

    xlabel('World X (m)', ...
        'FontSize', font_label, 'FontName', font_name);
    ylabel('World Y (m)', ...
        'FontSize', font_label, 'FontName', font_name);

    legend([h_poly, h_left, h_right, h_cop, h_start, h_end], ...
        {sprintf('Support polygon at %.2f s', t(kref)), ...
         'Left foot', 'Right foot', 'Actual CoP', ...
         'Start', 'End'}, ...
        'Location', 'southoutside', 'NumColumns', 2, ...
        'FontSize', font_legend, 'FontName', font_name);

    grid on;
    box on;
    axis equal;
    set(gca, 'FontSize', font_axis, 'LineWidth', 0.8, ...
        'FontName', font_name);

    % A square region in actual length units, encompassing all features.
    all_x = [poly_x; sole_x(:); xc(:)];
    all_y = [poly_y; sole_y(:); yc(:)];
    range_x = max(all_x) - min(all_x);
    range_y = max(all_y) - min(all_y);
    half_size = max([range_x, range_y, 0.02]) * 0.62;
    mid_x = 0.5 * (min(all_x) + max(all_x));
    mid_y = 0.5 * (min(all_y) + max(all_y));
    xlim(mid_x + [-half_size, half_size]);
    ylim(mid_y + [-half_size, half_size]);

    file_stem = output_names{fig_idx};
    exportgraphics(fig, fullfile(plot_dir, [file_stem '.png']), ...
        'Resolution', 300);
    savefig(fig, fullfile(plot_dir, [file_stem '.fig']));
    print(fig, fullfile(plot_dir, [file_stem '.pdf']), ...
        '-dpdf', '-painters');
end

% ------------------------------------------------------------
% FIGURE 4: INSTANTANEOUS SUPPORT MARGIN
% Positive: CoP is inside the corresponding instantaneous hull.
% Negative: CoP is outside the corresponding instantaneous hull.
% NaN: CoP or support hull is undefined.
% ------------------------------------------------------------
fig = figure('Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [5, 5, fig_width, fig_height]);
set(fig, 'PaperUnits', 'centimeters', ...
    'PaperSize', [fig_width, fig_height], ...
    'PaperPosition', [0, 0, fig_width, fig_height], ...
    'PaperPositionMode', 'manual');

plot(t, margin, 'r-', 'LineWidth', 1.2);
hold on;
yline(0, 'k--', 'LineWidth', 1.1);

xlabel('Time (s)', ...
    'FontSize', font_label, 'FontName', font_name);
ylabel('CoP support margin (m)', ...
    'FontSize', font_label, 'FontName', font_name);

legend('Actual CoP margin', 'Support boundary', ...
    'Location', 'best', 'FontSize', font_legend, ...
    'FontName', font_name);

grid on;
box on;
set(gca, 'FontSize', font_axis, 'LineWidth', 0.8, ...
    'FontName', font_name);
xlim([t(1), t(end)]);
ylim([-0.005, 0.055]); % Edit as desired.

file_stem = 'CoP_support_margin';
exportgraphics(fig, fullfile(plot_dir, [file_stem '.png']), ...
    'Resolution', 300);
savefig(fig, fullfile(plot_dir, [file_stem '.fig']));
print(fig, fullfile(plot_dir, [file_stem '.pdf']), ...
    '-dpdf', '-painters');

% ------------------------------------------------------------
% SUMMARY
% ------------------------------------------------------------
valid_margin = isfinite(margin);
fprintf('CSV: %s\n', csv_file);
fprintf('Logged samples: %d\n', numel(t));
fprintf('Valid CoP margin samples: %d\n', sum(valid_margin));
if any(valid_margin)
    fprintf('Minimum CoP support margin: %.6f m\n', ...
        min(margin(valid_margin)));
    fprintf('Samples outside the instantaneous hull: %d\n', ...
        sum(margin(valid_margin) < 0));
end
fprintf('Figure size: %.1f x %.1f cm\n', fig_width, fig_height);
fprintf('Saved plots to: %s\n', plot_dir);
