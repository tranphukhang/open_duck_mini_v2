clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_angles.csv');

data = readtable(csv_file);

t = data.time_s;

% Sign convention used in the paper figure:
% Reversed with respect to MuJoCo/XML:
%   Left : hip roll, hip pitch, knee, ankle
%   Right: hip roll, hip pitch

% Structure:
% {data, legend, y_min, y_max}

joint_data = {
     rad2deg(data.left_hip_yaw_rad),     'q_{L1}',  -0.1,   0.4;
    -rad2deg(data.left_hip_roll_rad),    'q_{L2}', -25.0,  20.0;
    -rad2deg(data.left_hip_pitch_rad),   'q_{L3}',  15.0,  40.0;
    -rad2deg(data.left_knee_rad),        'q_{L4}', -75.0, -45.0;
    -rad2deg(data.left_ankle_rad),       'q_{L5}',  25.0,  45.0;

     rad2deg(data.right_hip_yaw_rad),    'q_{R1}',  -0.25, -0.05;
    -rad2deg(data.right_hip_roll_rad),   'q_{R2}', -15.0,  30.0;
    -rad2deg(data.right_hip_pitch_rad),  'q_{R3}', -35.0, -15.0;
     rad2deg(data.right_knee_rad),       'q_{R4}',  50.0,  80.0;
     rad2deg(data.right_ankle_rad),      'q_{R5}', -45.0, -25.0
};

for i = 1:size(joint_data, 1)

    q = joint_data{i, 1};
    joint_label = joint_data{i, 2};

    y_min = joint_data{i, 3};
    y_max = joint_data{i, 4};

    figure('Color', 'w');

    plot(t, q, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', joint_label);

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
    ylim([y_min y_max]);

end