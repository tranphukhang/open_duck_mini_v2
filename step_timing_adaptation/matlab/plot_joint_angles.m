clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_angles.csv');

data = readtable(csv_file);

t = data.time_s;

near_limit_margin = 5; % deg

% Sign convention used in the paper figure:
% Reversed with respect to MuJoCo/XML:
%   Left : hip roll, hip pitch, knee, ankle
%   Right: hip roll, hip pitch
%
% If q_fig = -q_MJ, then the joint limits are transformed as
%   [q_min, q_max]_fig = [-q_max, -q_min]_MJ.

% Structure:
% {data, title, legend, lower_limit, upper_limit, y_min, y_max}

joint_data = {
     rad2deg(data.left_hip_yaw_rad),     'Left hip yaw',    'L1', -30,  30,  -0.1,   0.4;
    -rad2deg(data.left_hip_roll_rad),    'Left hip roll',   'L2', -25,  25, -35.0,  15.0;
    -rad2deg(data.left_hip_pitch_rad),   'Left hip pitch',  'L3', -30,  70,  10.0,  40.0;
    -rad2deg(data.left_knee_rad),        'Left knee',       'L4', -90,  90, -80.0, -50.0;
    -rad2deg(data.left_ankle_rad),       'Left ankle',      'L5', -90,  90,  20.0,  45.0;

     rad2deg(data.right_hip_yaw_rad),    'Right hip yaw',   'R1', -30,  30,  -0.25, -0.05;
    -rad2deg(data.right_hip_roll_rad),   'Right hip roll',  'R2', -25,  25, -20.0,  28.0;
    -rad2deg(data.right_hip_pitch_rad),  'Right hip pitch', 'R3', -70,  30, -40.0, -15.0;
     rad2deg(data.right_knee_rad),       'Right knee',      'R4', -90,  90,  50.0,  80.0;
     rad2deg(data.right_ankle_rad),      'Right ankle',     'R5', -90,  90, -45.0, -20.0
};

for i = 1:size(joint_data, 1)

    q = joint_data{i, 1};
    joint_title = joint_data{i, 2};
    joint_label = joint_data{i, 3};

    lower_limit = joint_data{i, 4};
    upper_limit = joint_data{i, 5};

    y_min = joint_data{i, 6};
    y_max = joint_data{i, 7};

    figure('Color', 'w');

    plot(t, q, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', joint_label);

    hold on;

    % Check whether the joint approaches or exceeds the limit
    lower_near = min(q) <= lower_limit + near_limit_margin;
    upper_near = max(q) >= upper_limit - near_limit_margin;

    limit_added = false;

    % Lower limit
    if lower_near

        yline(lower_limit, 'k--', ...
            'LineWidth', 1.3, ...
            'DisplayName', 'Limit');

        limit_added = true;

    end

    % Upper limit
    if upper_near

        if limit_added

            yline(upper_limit, 'k--', ...
                'LineWidth', 1.3, ...
                'HandleVisibility', 'off');

        else

            yline(upper_limit, 'k--', ...
                'LineWidth', 1.3, ...
                'DisplayName', 'Limit');

        end

    end

    xlabel('Time (s)', ...
        'FontSize', 12, ...
        'FontName', 'Times New Roman');

    ylabel('Amplitude (deg)', ...
        'FontSize', 12, ...
        'FontName', 'Times New Roman');

    title(joint_title, ...
        'FontSize', 13, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal');

    legend('Location', 'best', ...
        'FontSize', 11, ...
        'FontName', 'Times New Roman');

    grid on;
    box on;

    set(gca, ...
        'FontSize', 11, ...
        'LineWidth', 1.0, ...
        'FontName', 'Times New Roman');

    xlim([t(1) t(end)]);
    ylim([y_min y_max]);

end
