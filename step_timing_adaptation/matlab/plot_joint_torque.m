clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_torque.csv');

data = readtable(csv_file);

t = data.time_s;

%% =========================================================
% SIGN CONVENTION
% Convert MuJoCo/XML joint torques to the convention used
% in the MDH model and paper.
%
% Left:
%   L1 hip yaw   : same sign
%   L2 hip roll  : reversed
%   L3 hip pitch : reversed
%   L4 knee      : reversed
%   L5 ankle     : reversed
%
% Right:
%   R1 hip yaw   : same sign
%   R2 hip roll  : reversed
%   R3 hip pitch : reversed
%   R4 knee      : same sign
%   R5 ankle     : same sign
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

%% =========================================================
% JOINT TORQUE DATA
% Structure:
% {data, legend, y_min, y_max}
%% =========================================================

joint_data = {
    sL1*data.left_hip_yaw_Nm,      '\tau_{L1}', -0.1,  0.1;
    sL2*data.left_hip_roll_Nm,     '\tau_{L2}', -3.0,  4.0;
    sL3*data.left_hip_pitch_Nm,    '\tau_{L3}', -2.0,  2.5;
    sL4*data.left_knee_Nm,         '\tau_{L4}', -3.0,  6.0;
    sL5*data.left_ankle_Nm,        '\tau_{L5}', -2.0,  2.0;

    sR1*data.right_hip_yaw_Nm,     '\tau_{R1}', -0.1,  0.1;
    sR2*data.right_hip_roll_Nm,    '\tau_{R2}', -3.0,  4.0;
    sR3*data.right_hip_pitch_Nm,   '\tau_{R3}', -2.0,  2.5;
    sR4*data.right_knee_Nm,        '\tau_{R4}', -5.0,  4.2;
    sR5*data.right_ankle_Nm,       '\tau_{R5}', -2.0,  3.0
};

%% =========================================================
% PLOT
%% =========================================================

for i = 1:size(joint_data, 1)

    tau = joint_data{i, 1};
    joint_label = joint_data{i, 2};

    y_min = joint_data{i, 3};
    y_max = joint_data{i, 4};

    figure('Color', 'w');

    plot(t, tau, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', joint_label);

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
    ylim([y_min y_max]);

end