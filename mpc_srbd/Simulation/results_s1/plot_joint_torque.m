
clc;
clear;
close all;

% ============================================================
% JOINT TORQUE - STANDING S1 PUSH EXPERIMENT
% Two figures: Left leg and Right leg
% ============================================================

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

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

joint_names = {'hip_yaw', 'hip_roll', ...
    'hip_pitch', 'knee', 'ankle'};

foot_keys = {'L', 'R'};
foot_names = {'Left', 'Right'};

required_columns = {'time_s'};

for foot_idx = 1:2
    for joint_idx = 1:5
        required_columns{end+1} = sprintf( ...
            'tau_%s_%s_raw_Nm', ...
            foot_keys{foot_idx}, joint_names{joint_idx});
    end
end

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
% PLOT SETTINGS
% ============================================================

% Figure size (cm)
fig_width  = 8;
fig_height = 8;

% Font settings
font_name   = 'Times New Roman';
font_axis   = 10;
font_label  = 11;
font_legend = 8;

% Line colors
line_colors = [ ...
    0.00 0.00 0.00; ...  % L1/R1 - Black
    1.00 0.00 0.00; ...  % L2/R2 - Red
    0.00 0.447 0.741; ... % L3/R3 - Blue
    0.00 0.55 0.20; ...  % L4/R4 - Green
    0.55 0.20 0.70 ...   % L5/R5 - Purple
];

line_width = 1.2;

% Axis limits
x_limits = [t(1), t(end)];

% Y-axis limits (N.m)
% Use [] for automatic limits
y_limits = { ...
    [-0.4, 0.4], ... % Left leg
    [-0.4, 0.4]  ... % Right leg
};

% Output directory
plot_dir = fullfile(script_dir, 'Joint_torque_plots');

if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ============================================================
% PLOT JOINT TORQUE
% ============================================================

for foot_idx = 1:2

    foot_key = foot_keys{foot_idx};
    foot_name = foot_names{foot_idx};

    tau = zeros(numel(t), 5);
    legend_labels = cell(1, 5);

    % Read joint torques
    for joint_idx = 1:5

        column_name = sprintf( ...
            'tau_%s_%s_raw_Nm', ...
            foot_key, joint_names{joint_idx});

        tau(:, joint_idx) = data.(column_name);

        legend_labels{joint_idx} = sprintf( ...
            '%s%d', foot_key, joint_idx);
    end

    % Create figure
    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [5 5 fig_width fig_height]);

    set(fig, ...
        'PaperUnits', 'centimeters', ...
        'PaperSize', [fig_width fig_height], ...
        'PaperPosition', [0 0 fig_width fig_height]);

    hold on;

    % Store plot handles for legend
    h = gobjects(1, 5);

    % Plot five joint torques
    for joint_idx = 1:5

        h(joint_idx) = plot(t, tau(:, joint_idx), ...
            '-', ...
            'Color', line_colors(joint_idx, :), ...
            'LineWidth', line_width);
    end

    % X label
    xlabel('Time (s)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Y label
    ylabel('Joint torque (N.m)', ...
        'FontSize', font_label, ...
        'FontName', font_name);

    % Axes formatting
    grid on;
    box on;

    set(gca, ...
        'FontSize', font_axis, ...
        'LineWidth', 0.8, ...
        'FontName', font_name);

    % Axis limits
    xlim(x_limits);

    if ~isempty(y_limits{foot_idx})
        ylim(y_limits{foot_idx});
    end

    % ========================================================
    % LEGEND INSIDE PLOT
    % ========================================================

    lgd = legend(h, legend_labels, ...
        'Location', 'best', ...
        'NumColumns', 1, ...
        'FontSize', font_legend, ...
        'FontName', font_name);

    lgd.Box = 'on';
    lgd.Color = 'white';
    lgd.AutoUpdate = 'off';

    % ========================================================
    % SAVE FIGURES
    % ========================================================

    file_stem = sprintf( ...
        '%s_leg_joint_torque', foot_name);

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
