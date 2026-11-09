clc;
clear;
close all;

% ============================================================
% CONTACT FORCE COMPARISON
% MPC FORCE (BLACK) vs ACTUAL MUJOCO FORCE (RED)
% STANDING S1 PUSH EXPERIMENT
% ============================================================

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

% ============================================================
% IMPORT DATA
% ============================================================

required_columns = { ...
    'time_s', ...
    'actual_F_L_x_N', 'actual_F_L_y_N', 'actual_F_L_z_N', ...
    'actual_F_R_x_N', 'actual_F_R_y_N', 'actual_F_R_z_N', ...
    'mpc_F_L_x_N', 'mpc_F_L_y_N', 'mpc_F_L_z_N', ...
    'mpc_F_R_x_N', 'mpc_F_R_y_N', 'mpc_F_R_z_N'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

% ============================================================
% FORCE DATA
% ============================================================

% Left foot
actual_left = [ ...
    data.actual_F_L_x_N, ...
    data.actual_F_L_y_N, ...
    data.actual_F_L_z_N];

mpc_left = [ ...
    data.mpc_F_L_x_N, ...
    data.mpc_F_L_y_N, ...
    data.mpc_F_L_z_N];

% Right foot
actual_right = [ ...
    data.actual_F_R_x_N, ...
    data.actual_F_R_y_N, ...
    data.actual_F_R_z_N];

mpc_right = [ ...
    data.mpc_F_R_x_N, ...
    data.mpc_F_R_y_N, ...
    data.mpc_F_R_z_N];

% ============================================================
% PLOT SETTINGS
% ============================================================

axis_names = {'X', 'Y', 'Z'};
foot_names = {'Left', 'Right'};

% Y-axis limits for each foot and each axis
% Row 1: Left foot
% Row 2: Right foot
% Column 1: Fx, Column 2: Fy, Column 3: Fz

y_limits = { ...
    [-2.0,  1.2], [-3.4,  0.6], [-0.5, 14.0]; ... % Left foot
    [-1.8,  1.2], [-2.4,  0.6], [-0.5, 13.0]  ... % Right foot
};

x_limits = [t(1), t(end)];

% Figure size (cm)
fig_width  = 8;
fig_height = 8;

% Font settings
font_name   = 'Times New Roman';
font_axis   = 10;
font_label  = 11;
font_legend = 8;

% Line settings
mpc_color    = [0 0 0];   % Black
actual_color = [1 0 0];   % Red

% Output directory
plot_dir = fullfile(script_dir, 'Contact_force_compare_plots');

if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ============================================================
% PLOT LEFT FOOT
% ============================================================

for i = 1:3

    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 5 fig_width fig_height]);

    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    % MPC force - black
    plot(t, mpc_left(:, i), ...
        '-', ...
        'Color', mpc_color, ...
        'LineWidth', 1.5);

    hold on;

    % Actual force - red
    plot(t, actual_left(:, i), ...
        '-', ...
        'Color', actual_color, ...
        'LineWidth', 1.2);

    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    ylabel(sprintf('Left foot force %s (N)', axis_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    legend( ...
        'MPC force', ...
        'Actual force', ...
        'Location', 'best', ...
        'FontSize', font_legend, ...
        'FontName', font_name);

    grid on;
    box on;

    set(gca, ...
        'FontSize', font_axis, ...
        'LineWidth', 0.8, ...
        'FontName', font_name);

    xlim(x_limits);

    if ~isempty(y_limits{1, i})
        ylim(y_limits{1, i});
    end

    file_stem = sprintf('Left_foot_force_%s', lower(axis_names{i}));

    exportgraphics(fig, ...
        fullfile(plot_dir, [file_stem '.png']), ...
        'Resolution', 300);

    savefig(fig, ...
        fullfile(plot_dir, [file_stem '.fig']));

    print(fig, ...
        fullfile(plot_dir, [file_stem '.pdf']), ...
        '-dpdf', '-painters');
end

% ============================================================
% PLOT RIGHT FOOT
% ============================================================

for i = 1:3

    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 5 fig_width fig_height]);

    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    % MPC force - black
    plot(t, mpc_right(:, i), ...
        '-', ...
        'Color', mpc_color, ...
        'LineWidth', 1.5);

    hold on;

    % Actual force - red
    plot(t, actual_right(:, i), ...
        '-', ...
        'Color', actual_color, ...
        'LineWidth', 1.2);

    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    ylabel(sprintf('Right foot force %s (N)', axis_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    legend( ...
        'MPC force', ...
        'Actual force', ...
        'Location', 'best', ...
        'FontSize', font_legend, ...
        'FontName', font_name);

    grid on;
    box on;

    set(gca, ...
        'FontSize', font_axis, ...
        'LineWidth', 0.8, ...
        'FontName', font_name);

    xlim(x_limits);

    if ~isempty(y_limits{2, i})
        ylim(y_limits{2, i});
    end

    file_stem = sprintf('Right_foot_force_%s', lower(axis_names{i}));

    exportgraphics(fig, ...
        fullfile(plot_dir, [file_stem '.png']), ...
        'Resolution', 300);

    savefig(fig, ...
        fullfile(plot_dir, [file_stem '.fig']));

    print(fig, ...
        fullfile(plot_dir, [file_stem '.pdf']), ...
        '-dpdf', '-painters');
end

% ============================================================
% PRINT INFORMATION
% ============================================================

fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));
fprintf('Time range: %.3f to %.3f s\n', t(1), t(end));
fprintf('Figure size: %.1f x %.1f cm\n', fig_width, fig_height);
fprintf('Saved figures in: %s\n', plot_dir);