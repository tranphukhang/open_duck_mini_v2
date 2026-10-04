clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_torque.csv');

data = readtable(csv_file);

t = data.time_s;

torque_limit = 3.23;        % N m
near_limit_margin = 0.5;    % N m

% Cấu trúc:
% {data, title, legend, lower_limit, upper_limit, y_min, y_max}

joint_data = {
    data.left_hip_yaw_Nm,      'Mô-men chân trái hip yaw',     'L1', -torque_limit, torque_limit, -0.25,  0.25;
    data.left_hip_roll_Nm,     'Mô-men chân trái hip roll',    'L2', -torque_limit, torque_limit, -2.0,   2.5;
    data.left_hip_pitch_Nm,    'Mô-men chân trái hip pitch',   'L3', -torque_limit, torque_limit, -2.5,   2.0;
    data.left_knee_Nm,         'Mô-men chân trái knee',        'L4', -torque_limit, torque_limit, -4.5,   4.5;
    data.left_ankle_Nm,        'Mô-men chân trái ankle',       'L5', -torque_limit, torque_limit, -2.0,   2.0;

    data.right_hip_yaw_Nm,     'Mô-men chân phải hip yaw',     'R1', -torque_limit, torque_limit, -0.25,  0.25;
    data.right_hip_roll_Nm,    'Mô-men chân phải hip roll',    'R2', -torque_limit, torque_limit, -2.5,   2.5;
    data.right_hip_pitch_Nm,   'Mô-men chân phải hip pitch',   'R3', -torque_limit, torque_limit, -2.0,   2.0;
    data.right_knee_Nm,        'Mô-men chân phải knee',        'R4', -torque_limit, torque_limit, -4.5,   4.5;
    data.right_ankle_Nm,       'Mô-men chân phải ankle',       'R5', -torque_limit, torque_limit, -2.0,   2.0
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

    % Kiểm tra mô-men có gần hoặc vượt giới hạn hay không
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
        'FontSize', 12);

    ylabel('Amplitude (Nm)', ...
        'FontSize', 12);

    title(joint_title, ...
        'FontSize', 13, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal');

    legend('Location', 'best', ...
        'FontSize', 11);

    grid on;
    box on;

    set(gca, ...
        'FontSize', 11, ...
        'LineWidth', 1.0, ...
        'FontName', 'Times New Roman');

    xlim([t(1) t(end)]);
    ylim([y_min y_max]);

end