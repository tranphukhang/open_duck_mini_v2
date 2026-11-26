
clc;
clear;
close all;

% ============================================================
% FOOT CONTACT STATES - PPO EVALUATION
% ============================================================

%% LOAD DATA
script_dir = fileparts(mfilename('fullpath'));
log_dir = fullfile(script_dir, '..', 'log');

csv_candidates = dir(fullfile(log_dir, 'eval_*', 'states.csv'));

if numel(csv_candidates) == 1
    csv_file = fullfile(csv_candidates(1).folder, ...
        csv_candidates(1).name);

elseif numel(csv_candidates) > 1
    [~, newest_idx] = max([csv_candidates.datenum]);

    initial_file = fullfile(csv_candidates(newest_idx).folder, ...
        csv_candidates(newest_idx).name);

    [csv_name, csv_folder] = uigetfile( ...
        {'*.csv', 'CSV files (*.csv)'}, ...
        'Select states.csv', initial_file);

    if isequal(csv_name, 0)
        error('No states.csv file was selected.');
    end

    csv_file = fullfile(csv_folder, csv_name);
else
    error('No states.csv found inside: %s', log_dir);
end

%% IMPORT DATA
required_columns = { ...
    'time_s', 'contact_left', 'contact_right'};

opts = detectImportOptions(csv_file);

missing_columns = setdiff(required_columns, opts.VariableNames);

if ~isempty(missing_columns)
    error('Missing CSV columns: %s', ...
        strjoin(missing_columns, ', '));
end

opts.SelectedVariableNames = required_columns;
data = readtable(csv_file, opts);

t = data.time_s;

contact = [ ...
    data.contact_left, ...
    data.contact_right];

%% PLOT SETTINGS

foot_names = {'Left Foot', 'Right Foot'};

fig_width  = 8;
fig_height = 8;

font_name   = 'Times New Roman';
font_axis   = 10;
font_label  = 11;
font_legend = 8;

x_limits = [min(t), 3];
y_limits = [-0.5, 1.5];

%% PLOT FOOT CONTACTS

for i = 1:2

    fig = figure( ...
        'Name', [foot_names{i} ' Contact'], ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 + 9*(i-1), 5, fig_width, fig_height]);

    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width, fig_height], ...
        'PaperPosition', [0, 0, fig_width, fig_height]);

    % Actual contact state - Red
    line_colors = {'b-', 'r-'};

    stairs(t, contact(:, i), ...
        line_colors{i}, 'LineWidth', 1.2);

    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    ylabel('Contact State', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    legend('Contact', ...
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
    ylim(y_limits);
    yticks([0, 1]);

end

fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));
