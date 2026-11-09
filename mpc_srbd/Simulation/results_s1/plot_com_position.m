
clc;
clear;
close all;

% ============================================================
% CoM POSITION TRACKING - STANDING S1 PUSH EXPERIMENT
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
    'com_x_m', 'com_y_m', 'com_z_m', ...
    'ref_com_x_m', 'ref_com_y_m', 'ref_com_z_m'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

com_actual = [ ...
    data.com_x_m, ...
    data.com_y_m, ...
    data.com_z_m];

com_desired = [ ...
    data.ref_com_x_m, ...
    data.ref_com_y_m, ...
    data.ref_com_z_m];

% ============================================================
% PLOT SETTINGS
% ============================================================

axis_names = {'X', 'Y', 'Z'};

% Y-axis limits
% Use [] for automatic limits

y_limits = { ...
    [-0.020, 0.000], ...  % CoM X
    [-0.002, 0.014], ...  % CoM Y
    [ 0.174, 0.185]  ...  % CoM Z
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
plot_dir = fullfile(script_dir, 'CoM_position_plots');

if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ============================================================
% PLOT CoM POSITION
% ============================================================

for i = 1:3

    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 5 fig_width fig_height]);

    % Set physical export dimensions
    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    % Desired CoM position
    plot(t, com_desired(:, i), ...
        'k-', 'LineWidth', 1.5);

    hold on;

    % Actual CoM position
    plot(t, com_actual(:, i), ...
        'r-', 'LineWidth', 1.2);

    % X label
    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Y label
    ylabel(sprintf('CoM %s position (m)', axis_names{i}), ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Legend
    legend( ...
        'Desired CoM position', ...
        'Actual CoM position', ...
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

    % X-axis limit
    xlim(x_limits);

    % Y-axis limit
    if ~isempty(y_limits{i})
        ylim(y_limits{i});
    end

    % ========================================================
    % SAVE FIGURE
    % ========================================================

    file_stem = sprintf( ...
        'CoM_position_%s', lower(axis_names{i}));

    % PNG, 300 DPI
    exportgraphics(fig, ...
        fullfile(plot_dir, [file_stem '.png']), ...
        'Resolution', 300);

    % Editable MATLAB figure
    savefig(fig, ...
        fullfile(plot_dir, [file_stem '.fig']));

    % Vector PDF with 8 x 8 cm page
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
