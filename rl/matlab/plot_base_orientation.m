clc;
clear;
close all;

% ============================================================
% BASE ORIENTATION - ROLL, PITCH, YAW (ONE FIGURE)
% ============================================================

%% LOAD DATA
script_dir = fileparts(mfilename('fullpath'));
log_dir = fullfile(script_dir, '..', 'log');

csv_candidates = dir(fullfile(log_dir, 'eval_*', 'states.csv'));

if numel(csv_candidates) == 1
    csv_file = fullfile(csv_candidates(1).folder, ...
        csv_candidates(1).name);
elseif numel(csv_candidates) > 1
    % Automatically select the newest simulation result.
    [~, idx] = max([csv_candidates.datenum]);
    csv_file = fullfile(csv_candidates(idx).folder, ...
        csv_candidates(idx).name);
else
    [csv_name, csv_folder] = uigetfile( ...
        {'*.csv', 'CSV files (*.csv)'}, ...
        'Select states.csv', log_dir);
    if isequal(csv_name, 0)
        error('No states.csv file was selected.');
    end
    csv_file = fullfile(csv_folder, csv_name);
end

%% IMPORT DATA
required_columns = {'time_s', 'roll_deg', 'pitch_deg', 'yaw_deg'};
opts = detectImportOptions(csv_file);
missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;
roll_deg  = data.roll_deg;
pitch_deg = data.pitch_deg;
yaw_deg   = data.yaw_deg;

if isempty(t)
    error('The CSV file contains no samples.');
end

%% PLOT SETTINGS
% Y-axis limits (deg); [] = automatic limits
y_limits = [];  % Example: [-15, 15]
x_limits = [min(t), max(t)];

% Figure size (cm)
fig_width  = 8;
fig_height = 8;

% Font settings
font_name   = 'Times New Roman';
font_axis   = 10;
font_label  = 11;
font_legend = 8;

%% PLOT ALL ORIENTATION ANGLES IN ONE FIGURE
fig = figure( ...
    'Name', 'Base Orientation', ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [5, 5, fig_width, fig_height]);

set(fig, ...
    'PaperUnits', 'centimeters', ...
    'PaperSize', [fig_width, fig_height], ...
    'PaperPosition', [0, 0, fig_width, fig_height]);

plot(t, roll_deg,  'r-', 'LineWidth', 1.2);
hold on;
plot(t, pitch_deg, 'b-', 'LineWidth', 1.2);
plot(t, yaw_deg,   'k-', 'LineWidth', 1.2);
hold off;

xlabel('Time (s)', ...
    'FontSize', font_label, 'FontName', font_name);
ylabel('Orientation (deg)', ...
    'FontSize', font_label, 'FontName', font_name);

legend({'Roll', 'Pitch', 'Yaw'}, ...
    'Location', 'best', ...
    'FontSize', font_legend, ...
    'FontName', font_name);

grid on;
box on;

set(gca, ...
    'FontSize', font_axis, ...
    'LineWidth', 0.8, ...
    'FontName', font_name);

if diff(x_limits) > 0
    xlim(x_limits);
end
if ~isempty(y_limits)
    ylim(y_limits);
end

%% PRINT INFORMATION
fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));
fprintf('Time range: %.3f to %.3f s\n', t(1), t(end));
