clc;
clear;
close all;

%% =========================================================
% JOINT TORQUES - LEFT/RIGHT PAIRS
%
% Create 5 separate figures:
%   Figure 1: tau_L1 and tau_R1  (Hip yaw)
%   Figure 2: tau_L2 and tau_R2  (Hip roll)
%   Figure 3: tau_L3 and tau_R3  (Hip pitch)
%   Figure 4: tau_L4 and tau_R4  (Knee)
%   Figure 5: tau_L5 and tau_R5  (Ankle)
%
% Sign convention is identical to the original
% plot_joint_torque.m file.
%% =========================================================

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_torque.csv');

data = readtable(csv_file);

t = data.time_s;

%% =========================================================
% SIGN CONVENTION
% Convert MuJoCo/XML joint torques to the convention used
% in the MDH model and paper.
%% =========================================================

sL1 =  1;
sL2 = -1;
sL3 = -1;
sL4 = -1;
sL5 = -1;

sR1 =  1;
sR2 = -1;
sR3 = -1;
sR4 =  1;
sR5 =  1;

tauL = {
    sL1 * data.left_hip_yaw_Nm
    sL2 * data.left_hip_roll_Nm
    sL3 * data.left_hip_pitch_Nm
    sL4 * data.left_knee_Nm
    sL5 * data.left_ankle_Nm
};

tauR = {
    sR1 * data.right_hip_yaw_Nm
    sR2 * data.right_hip_roll_Nm
    sR3 * data.right_hip_pitch_Nm
    sR4 * data.right_knee_Nm
    sR5 * data.right_ankle_Nm
};

left_label = {
    '\tau_{L1}'
    '\tau_{L2}'
    '\tau_{L3}'
    '\tau_{L4}'
    '\tau_{L5}'
};

right_label = {
    '\tau_{R1}'
    '\tau_{R2}'
    '\tau_{R3}'
    '\tau_{R4}'
    '\tau_{R5}'
};

joint_name = {
    'Hip yaw'
    'Hip roll'
    'Hip pitch'
    'Knee'
    'Ankle'
};

%% =========================================================
% SHARED Y-LIMITS FOR EACH LEFT/RIGHT PAIR
%
% Obtained by combining the original left/right limits.
%% =========================================================

y_lim = [
    -0.10   0.10
    -3.00   5.00
    -2.00   3.00
    -5.00   7.00
    -2.00   3.00
];

%% =========================================================
% ACTUATOR LIMIT
%
% STS3215 torque limit used in the paper.
% Uncomment the yline commands below if you want the
% actuator limits shown on each plot.
%% =========================================================

tau_limit = 3.25;   % N.m

%% =========================================================
% PLOT
%% =========================================================

for i = 1:5

    figure( ...
        'Color', 'w', ...
        'Position', [100 100 650 480]);

    hold on;

    plot(t, tauL{i}, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', left_label{i});

    plot(t, tauR{i}, 'b--', ...
        'LineWidth', 1.5, ...
        'DisplayName', right_label{i});

    % Optional actuator limits:
    % yline(tau_limit, 'k--', ...
    %     'LineWidth', 1.2, ...
    %     'HandleVisibility', 'off');
    %
    % yline(-tau_limit, 'k--', ...
    %     'LineWidth', 1.2, ...
    %     'HandleVisibility', 'off');

    xlabel('Time (s)', ...
        'FontSize', 24, ...
        'FontName', 'Times New Roman');

    ylabel('Amplitude (N.m)', ...
        'FontSize', 24, ...
        'FontName', 'Times New Roman');

    legend('Location', 'best', ...
        'FontSize', 16, ...
        'FontName', 'Times New Roman', ...
        'Interpreter', 'tex');

    grid on;
    box on;

    set(gca, ...
        'FontSize', 22, ...
        'LineWidth', 1.0, ...
        'FontName', 'Times New Roman');

    xlim([t(1) t(end)]);
    ylim(y_lim(i, :));

    % Optional title:
    % title(joint_name{i}, ...
    %     'FontSize', 22, ...
    %     'FontName', 'Times New Roman', ...
    %     'FontWeight', 'normal');

end
