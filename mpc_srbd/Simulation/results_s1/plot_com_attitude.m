
clc;
clear;
close all;

% ============================================================
% ROLL, PITCH, YAW TRACKING - STANDING S1 PUSH EXPERIMENT
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
    'roll_rad', 'pitch_rad', 'yaw_rad', ...
    'ref_roll_rad', 'ref_pitch_rad', 'ref_yaw_rad'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

% Convert radians to degrees
angle_actual_deg = rad2deg([ ...
    data.roll_rad, ...
    data.pitch_rad, ...
    data.yaw_rad]);

angle_desired_deg = rad2deg([ ...
    data.ref_roll_rad, ...
    data.ref_pitch_rad, ...
    data.ref_yaw_rad]);

% ============================================================
% PLOT SETTINGS
% ============================================================

angle_names = {'Roll', 'Pitch', 'Yaw'};

% Y-axis limits (deg)
% Use [] for automatic limits

y_limits = { ...
    [-1.0, 1.0], ...  % Roll
    [-2.0, 2.0], ...  % Pitch
    [-0.6, 0.3]  ...  % Yaw
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
plot_dir = fullfile(script_dir, 'CoM_attitude_plots');

if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ============================================================
% PLOT ROLL, PITCH, YAW
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

    % Desired angle
    plot(t, angle_desired_deg(:, i), ...
        'k-', 'LineWidth', 1.5);

    hold on;

    % Actual angle
    plot(t, angle_actual_deg(:, i), ...
        'r-', 'LineWidth', 1.2);

    % X label
    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Y label
    ylabel(sprintf('%s angle (deg)', angle_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Legend
    legend( ...
        'Desired angle', ...
        'Actual angle', ...
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
        'CoM_attitude_%s', lower(angle_names{i}));

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
