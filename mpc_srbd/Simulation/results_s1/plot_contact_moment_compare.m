
clc;
clear;
close all;

% ============================================================
% CONTACT MOMENT COMPARISON - STANDING S1 PUSH EXPERIMENT
% MPC MOMENT (BLACK) vs MUJOCO CONTACT MOMENT (RED)
%
% Each figure: one foot and one moment component (Mx, My, Mz).
% All moments are expressed in World and referenced to L6/R6.
% Figure size: 8 x 8 cm.
% ============================================================

% ------------------------------------------------------------
% SELECT CSV FILE
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
% IMPORT DATA
% ------------------------------------------------------------
opts = detectImportOptions(csv_file);
data = readtable(csv_file, opts);

t = data.time_s;

% IMPORTANT:
% MuJoCo actual_M_*_origin_* moments are about O=(0,0,ground_z).
% MPC mpc_M_*_* moments are about the L6/R6 foot sites.
% Shift actual moments to the corresponding foot site:
%
% M_site = M_origin - cross(p_site - p_origin, F_actual)
%
% Both sides are then in World coordinates about the SAME point.

% Ground plane height from scene_flat_terrian_v3.xml
ground_z_m = 0.0;

% ------------------------------------------------------------
% PLOT SETTINGS
% ------------------------------------------------------------
foot_keys  = {'L', 'R'};
foot_names = {'Left', 'Right'};
axis_names = {'X', 'Y', 'Z'};
axis_keys  = {'x', 'y', 'z'};

% Y-axis limits in N.m:
% Row 1 = Left foot, Row 2 = Right foot
% Columns = Mx, My, Mz
% Use [] for automatic limits.
y_limits = { ...
    [-0.07, 0.20], [-0.50, 0.22], [-0.085, 0.035]; ... % Left
    [-0.15, 0.20], [-0.46, 0.22], [-0.050, 0.035]  ... % Right
};

x_limits = [t(1), t(end)];

fig_width  = 8;  % cm
fig_height = 8;  % cm

font_name   = 'Times New Roman';
font_axis   = 10;
font_label  = 11;
font_legend = 8;

mpc_color    = [0 0 0];  % Black
actual_color = [1 0 0];  % Red

plot_dir = fullfile(script_dir, 'Contact_moment_compare_plots');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

% ------------------------------------------------------------
% PLOT EACH FOOT AND EACH COMPONENT (6 FIGURES)
% ------------------------------------------------------------
for foot_idx = 1:2

    foot_key = foot_keys{foot_idx};
    foot_name = foot_names{foot_idx};

    % Preallocate vectors [N x 3]
    moment_mpc       = zeros(numel(t), 3);
    moment_origin    = zeros(numel(t), 3);
    force_actual     = zeros(numel(t), 3);
    foot_site        = zeros(numel(t), 3);

    for axis_idx = 1:3
        a = axis_keys{axis_idx};

        col_mpc = sprintf('mpc_M_%s_%s_Nm', foot_key, a);
        col_actual = sprintf( ...
            'actual_M_%s_origin_%s_Nm', foot_key, a);
        col_force = sprintf('actual_F_%s_%s_N', foot_key, a);
        col_site = sprintf('foot_site_%s_%s_m', foot_key, a);

        names = {col_mpc, col_actual, col_force, col_site};
        missing = setdiff(names, data.Properties.VariableNames);

        if ~isempty(missing)
            error('Missing CSV columns: %s', strjoin(missing, ', '));
        end

        moment_mpc(:, axis_idx)    = data.(col_mpc);
        moment_origin(:, axis_idx) = data.(col_actual);
        force_actual(:, axis_idx)  = data.(col_force);
        foot_site(:, axis_idx)     = data.(col_site);
    end

    % Convert actual contact moments to the L6/R6 origin.
    arm = foot_site;
    arm(:, 3) = arm(:, 3) - ground_z_m;
    moment_actual = moment_origin - cross(arm, force_actual, 2);

    for axis_idx = 1:3

        fig = figure( ...
            'Color', 'w', ...
            'Units', 'centimeters', ...
            'Position', [5 5 fig_width fig_height]);

        set(fig, ...
            'PaperUnits', 'centimeters', ...
            'PaperSize', [fig_width fig_height], ...
            'PaperPosition', [0 0 fig_width fig_height], ...
            'PaperPositionMode', 'manual');

        % MPC contact moment - black
        plot(t, moment_mpc(:, axis_idx), ...
            '-', 'Color', mpc_color, 'LineWidth', 1.5);
        hold on;

        % Actual MuJoCo contact moment - red
        plot(t, moment_actual(:, axis_idx), ...
            '-', 'Color', actual_color, 'LineWidth', 1.2);

        xlabel('Time (s)', ...
            'FontSize', font_label, 'FontName', font_name);

        ylabel(sprintf('%s foot moment %s (N.m)', ...
            foot_name, axis_names{axis_idx}), ...
            'FontSize', font_label, 'FontName', font_name);

        legend('MPC moment', 'Actual moment', ...
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

        if ~isempty(y_limits{foot_idx, axis_idx})
            ylim(y_limits{foot_idx, axis_idx});
        end

        % Save PNG, FIG, PDF
        file_stem = sprintf('%s_foot_moment_%s', ...
            foot_name, lower(axis_names{axis_idx}));

        exportgraphics(fig, ...
            fullfile(plot_dir, [file_stem '.png']), ...
            'Resolution', 300);

        savefig(fig, ...
            fullfile(plot_dir, [file_stem '.fig']));

        print(fig, ...
            fullfile(plot_dir, [file_stem '.pdf']), ...
            '-dpdf', '-painters');
    end
end

% ------------------------------------------------------------
% PRINT INFORMATION
% ------------------------------------------------------------
fprintf('Loaded: %s\n', csv_file);
fprintf('Samples: %d\n', numel(t));
fprintf('Time range: %.3f to %.3f s\n', t(1), t(end));
fprintf('Figure size: %.1f x %.1f cm\n', fig_width, fig_height);
fprintf('Actual moments shifted to L6/R6 reference sites.\n');
fprintf('Saved figures in: %s\n', plot_dir);
