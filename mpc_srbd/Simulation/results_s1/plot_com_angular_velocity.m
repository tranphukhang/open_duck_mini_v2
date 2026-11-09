
clc;
clear;
close all;

% ============================================================
% CoM ANGULAR VELOCITY - STANDING S1 PUSH EXPERIMENT
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
    'omega_x_rad_s', ...
    'omega_y_rad_s', 'omega_z_rad_s'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

% Actual angular velocity (rad/s)
angular_velocity_actual = [ ...
    data.omega_x_rad_s, ...
    data.omega_y_rad_s, ...
    data.omega_z_rad_s];

% Standing S1: desired angular velocity is zero
angular_velocity_desired = zeros(numel(t), 3);

% ============================================================
% PLOT SETTINGS
% ============================================================

axis_names = {'X', 'Y', 'Z'};

% Y-axis limits (rad/s)
% Use [] for automatic limits

y_limits = { ...
    [-0.22, 0.4], ...   % Angular velocity X
    [-0.36, 0.7], ...   % Angular velocity Y
    [-0.055, 0.035] ...  % Angular velocity Z
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

% Output directory
plot_dir = fullfile(script_dir, 'CoM_angular_velocity_plots');

if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ============================================================
% PLOT CoM ANGULAR VELOCITY
% ============================================================

for i = 1:3

    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 5 fig_width fig_height]);

    % Physical dimensions
    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    % Desired angular velocity
    plot(t, angular_velocity_desired(:, i), ...
        'k-', 'LineWidth', 1.5);

    hold on;

    % Actual angular velocity
    plot(t, angular_velocity_actual(:, i), ...
        'r-', 'LineWidth', 1.2);

    % X label
    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Y label
    ylabel(sprintf('Angular velocity %s (rad/s)', ...
        axis_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Legend
    legend( ...
        'Desired angular velocity', ...
        'Actual angular velocity', ...
        'Location', 'best', ...
        'FontSize', font_legend, ...
        'FontName', font_name);

    grid on;
    box on;

    % Axes formatting
    set(gca, ...
        'FontSize', font_axis, ...
        'LineWidth', 0.8, ...
        'FontName', font_name);

    % X-axis limits
    xlim(x_limits);

    % Y-axis limits
    if ~isempty(y_limits{i})
        ylim(y_limits{i});
    end

    % ========================================================
    % SAVE FIGURES
    % ========================================================

    file_stem = sprintf( ...
        'CoM_angular_velocity_%s', lower(axis_names{i}));

    % PNG - 300 DPI
    exportgraphics(fig, ...
        fullfile(plot_dir, [file_stem '.png']), ...
        'Resolution', 300);

    % Editable MATLAB figure
    savefig(fig, ...
        fullfile(plot_dir, [file_stem '.fig']));

    % Vector PDF - 8 x 8 cm
    print(fig, ...
        fullfile(plot_dir, [file_stem '.pdf']), ...
        '-dpdf', '-painters');

end

% ============================================================
% PRINT INFORMATION
% ============================================================

fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));

fprintf('Time range: %.3f to %.3f s\n', ...
    t(1), t(end));

fprintf('Figure size: %.1f x %.1f cm\n', ...
    fig_width, fig_height);

fprintf('Saved figures in: %s\n', plot_dir);
