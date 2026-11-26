clc;
clear;
close all;

% ============================================================
% BASE ANGULAR VELOCITY - THREE SEPARATE FIGURES
% ============================================================

%% LOAD DATA
script_dir = fileparts(mfilename('fullpath'));
log_dir = fullfile(script_dir, '..', 'log');
csv_candidates = dir(fullfile(log_dir, 'eval_*', 'states.csv'));

if numel(csv_candidates) == 1
    csv_file = fullfile(csv_candidates(1).folder, csv_candidates(1).name);
else
    if ~isempty(csv_candidates)
        [~, newest_idx] = max([csv_candidates.datenum]);
        initial_path = fullfile(csv_candidates(newest_idx).folder, ...
            csv_candidates(newest_idx).name);
    else
        initial_path = log_dir;
    end

    [csv_name, csv_folder] = uigetfile( ...
        {'*.csv', 'CSV files (*.csv)'}, 'Select states.csv', initial_path);
    if isequal(csv_name, 0)
        error('No states.csv file was selected.');
    end
    csv_file = fullfile(csv_folder, csv_name);
end

%% IMPORT DATA
required_columns = {'time_s', 'wx_rad_s', 'wy_rad_s', 'wz_rad_s'};
opts = detectImportOptions(csv_file);
missing_columns = setdiff(required_columns, opts.VariableNames);
if ~isempty(missing_columns)
    error('Missing CSV columns: %s', strjoin(missing_columns, ', '));
end
opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;
w_actual = [data.wx_rad_s, data.wy_rad_s, data.wz_rad_s];

%% PLOT SETTINGS
axis_names = {'X', 'Y', 'Z'};

% Y-axis limits (rad/s); [] = automatic
% Set each figure's limits independently.
y_limits = { ...
    [], ...  % wx
    [], ...  % wy
    []  ...  % wz
};

fig_width = 8;
fig_height = 8;
font_name = 'Times New Roman';
font_axis = 10;
font_label = 11;
font_legend = 8;

%% PLOT
for i = 1:3
    fig = figure( ...
        'Name', sprintf('Angular Velocity %s', axis_names{i}), ...
        'Color', 'w', 'Units', 'centimeters', ...
        'Position', [5 + 9*(i-1), 5, fig_width, fig_height]);

    set(fig, 'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width, fig_height], ...
        'PaperPosition', [0, 0, fig_width, fig_height]);

    hold on;
    if i == 3
        % Only yaw angular velocity has the zero desired reference.
        plot(t, zeros(size(t)), 'k--', 'LineWidth', 1.5);
    end

    plot(t, w_actual(:, i), 'r-', 'LineWidth', 1.2);
    hold off;

    xlabel('Time (s)', 'FontName', font_name, ...
        'FontSize', font_label);
    ylabel(sprintf('Angular Velocity %s (rad/s)', axis_names{i}), ...
        'FontName', font_name, 'FontSize', font_label);

    if i == 3
        legend('Desired', 'Actual', 'Location', 'best', ...
            'FontName', font_name, 'FontSize', font_legend);
    else
        legend('Actual', 'Location', 'best', ...
            'FontName', font_name, 'FontSize', font_legend);
    end

    grid on;
    box on;
    set(gca, 'FontName', font_name, 'FontSize', font_axis, ...
        'LineWidth', 0.8);
    xlim([min(t), max(t)]);
    if ~isempty(y_limits{i})
        ylim(y_limits{i});
    end
end

fprintf('Loaded: %s\n', csv_file);
