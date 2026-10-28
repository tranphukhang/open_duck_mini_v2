clc;
clear;
close all;

%% =========================================================
% JOINT ANGLES - LEFT/RIGHT PAIRS
%
% Create 5 separate figures:
%   Figure 1: q_L1 and q_R1  (Hip yaw)
%   Figure 2: q_L2 and q_R2  (Hip roll)
%   Figure 3: q_L3 and q_R3  (Hip pitch)
%   Figure 4: q_L4 and q_R4  (Knee)
%   Figure 5: q_L5 and q_R5  (Ankle)
%
% Sign convention is identical to the original
% plot_joint_angles.m file.
%% =========================================================

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_angles.csv');

data = readtable(csv_file);

t = data.time_s;

%% =========================================================
% SIGN CONVENTION
% Same convention used in the paper figure
%% =========================================================

qL = {
     rad2deg(data.left_hip_yaw_rad)
    -rad2deg(data.left_hip_roll_rad)
    -rad2deg(data.left_hip_pitch_rad)
    -rad2deg(data.left_knee_rad)
    -rad2deg(data.left_ankle_rad)
};

qR = {
     rad2deg(data.right_hip_yaw_rad)
    -rad2deg(data.right_hip_roll_rad)
    -rad2deg(data.right_hip_pitch_rad)
     rad2deg(data.right_knee_rad)
     rad2deg(data.right_ankle_rad)
};

left_label = {
    'q_{L1}'
    'q_{L2}'
    'q_{L3}'
    'q_{L4}'
    'q_{L5}'
};

right_label = {
    'q_{R1}'
    'q_{R2}'
    'q_{R3}'
    'q_{R4}'
    'q_{R5}'
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
% Obtained by combining the y-limits of the original
% individual left and right joint plots.
%% =========================================================

y_lim = [
    -0.25    0.40
   -25.00   40.00
   -35.00   40.00
   -75.00   80.00
   -45.00   45.00
];

%% =========================================================
% PLOT
%% =========================================================

for i = 1:5

    figure( ...
        'Color', 'w', ...
        'Position', [100 100 650 480]);

    hold on;

    plot(t, qL{i}, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', left_label{i});

    plot(t, qR{i}, 'b--', ...
        'LineWidth', 1.5, ...
        'DisplayName', right_label{i});

    xlabel('Time (s)', ...
        'FontSize', 24, ...
        'FontName', 'Times New Roman');

    ylabel('Amplitude (degree)', ...
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
