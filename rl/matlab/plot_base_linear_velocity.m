
clc;
clear;
close all;

% ============================================================
% BASE LINEAR VELOCITY - PPO EVALUATION
% ============================================================

script_dir = fileparts(mfilename('fullpath'));
log_dir = fullfile(script_dir, '..', 'log');

% Find states.csv in evaluation folders
csv_candidates = dir(fullfile(log_dir, 'eval_*', 'states.csv'));

if numel(csv_candidates) == 1
    csv_file = fullfile(csv_candidates(1).folder, ...
        csv_candidates(1).name);
else
    [csv_name, csv_folder] = uigetfile( ...
        {'*.csv', 'CSV files (*.csv)'}, ...
        'Select states.csv', log_dir);

    if isequal(csv_name, 0)
        error('No states.csv file was selected.');
    end

    csv_file = fullfile(csv_folder, csv_name);
end

% ============================================================
% IMPORT DATA
% ============================================================

required_columns = { ...
    'time_s', ...
    'vx_desired_m_s', 'vx_actual_m_s', ...
    'vy_desired_m_s', 'vy_actual_m_s'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

% Desired linear velocity (m/s)
velocity_desired = [ ...
    data.vx_desired_m_s, ...
    data.vy_desired_m_s];

% Actual linear velocity (m/s)
velocity_actual = [ ...
    data.vx_actual_m_s, ...
    data.vy_actual_m_s];

% ============================================================
% PLOT SETTINGS
% ============================================================

axis_names = {'X', 'Y'};

% Y-axis limits (m/s)
% Use [] for automatic limits
y_limits = { ...
    [], ...  % Linear velocity X
    [-0.4, 0.4]  ...  % Linear velocity Y
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

% ============================================================
% PLOT BASE LINEAR VELOCITY
% ============================================================

for i = 1:2

    fig = figure( ...
        'Name', sprintf('Base Linear Velocity %s', axis_names{i}), ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 + 9*(i-1), 5, fig_width, fig_height]);

    % Physical dimensions
    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    % Desired velocity - Black
    plot(t, velocity_desired(:, i), ...
        'k-', 'LineWidth', 1.5);

    hold on;

    % Actual velocity - Red
    plot(t, velocity_actual(:, i), ...
        'r-', 'LineWidth', 1.2);

    % X-axis label
    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Y-axis label
    ylabel(sprintf('Linear Velocity %s (m/s)', axis_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Legend
    legend( ...
        'Desired Velocity', ...
        'Actual Velocity', ...
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

    % Axis limits
    xlim(x_limits);

    if ~isempty(y_limits{i})
        ylim(y_limits{i});
    end

end

% ============================================================
% PRINT INFORMATION
% ============================================================

fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));
fprintf('Time range: %.3f to %.3f s\n', t(1), t(end));
