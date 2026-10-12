clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_torque.csv');

data = readtable(csv_file);

t = data.time_s;

torque_limit = 3.23;        % N.m
near_limit_margin = 0.5;    % N.m

% Sign convention used in the paper figure:
% Reversed with respect to MuJoCo/XML:
%   Left : hip roll, hip pitch, knee, ankle
%   Right: hip roll, hip pitch
%
% If q_fig = -q_MJ, the corresponding generalized torque also changes sign:
%   tau_fig = -tau_MJ.
% The actuator torque limits remain [-3.23, 3.23] N.m because they are symmetric.

% Structure:
% {data, title, legend, lower_limit, upper_limit, y_min, y_max}

joint_data = {
     data.left_hip_yaw_Nm,      'Left hip yaw torque',      'L1', -torque_limit, torque_limit, -0.12,   0.1;
    -data.left_hip_roll_Nm,     'Left hip roll torque',     'L2', -torque_limit, torque_limit, -5.0,   3.0;
    -data.left_hip_pitch_Nm,    'Left hip pitch torque',    'L3', -torque_limit, torque_limit, -2.0,   2.5;
    -data.left_knee_Nm,         'Left knee torque',         'L4', -torque_limit, torque_limit, -5.5,   5.0;
    -data.left_ankle_Nm,        'Left ankle torque',        'L5', -torque_limit, torque_limit, -3.0,   2.0;

     data.right_hip_yaw_Nm,     'Right hip yaw torque',     'R1', -torque_limit, torque_limit, -0.1,   0.15;
    -data.right_hip_roll_Nm,    'Right hip roll torque',    'R2', -torque_limit, torque_limit, -4.0,   3.5;
    -data.right_hip_pitch_Nm,   'Right hip pitch torque',   'R3', -torque_limit, torque_limit, -3.0,   2.0;
     data.right_knee_Nm,        'Right knee torque',        'R4', -torque_limit, torque_limit, -5.0,   5.5;
     data.right_ankle_Nm,       'Right ankle torque',       'R5', -torque_limit, torque_limit, -2.0,   3.0
};

for i = 1:size(joint_data, 1)

    tau = joint_data{i, 1};
    joint_title = joint_data{i, 2};
    joint_label = joint_data{i, 3};

    lower_limit = joint_data{i, 4};
    upper_limit = joint_data{i, 5};

    y_min = joint_data{i, 6};
    y_max = joint_data{i, 7};

    figure('Color', 'w');

    plot(t, tau, 'r-', ...
        'LineWidth', 1.5, ...
        'DisplayName', joint_label);

    hold on;

    % Check whether the torque approaches or exceeds the limit
    lower_near = min(tau) <= lower_limit + near_limit_margin;
    upper_near = max(tau) >= upper_limit - near_limit_margin;

    limit_added = false;

    % Lower torque limit
    if lower_near

        yline(lower_limit, 'k--', ...
            'LineWidth', 1.3, ...
            'DisplayName', 'Limit');

        limit_added = true;

    end

    % Upper torque limit
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

    ylabel('Amplitude (N.m)', ...
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
