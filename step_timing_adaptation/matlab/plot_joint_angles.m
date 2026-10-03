clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'joint_angles.csv');

data = readtable(csv_file);

t = data.time_s;

near_limit_margin = 5; % deg

% Cấu trúc:
% {data, title, legend, lower_limit, upper_limit, y_min, y_max}

joint_data = {
    rad2deg(data.left_hip_yaw_rad),     'Chân trái hip yaw',     'L1', -30,  30, -0.5,  0.5;
    rad2deg(data.left_hip_roll_rad),    'Chân trái hip roll',    'L2', -25,  25, 0,  30;
    rad2deg(data.left_hip_pitch_rad),   'Chân trái hip pitch',   'L3', -70,  30, -50,  -20;
    rad2deg(data.left_knee_rad),        'Chân trái knee',        'L4', -90,  90, 60, 100;
    rad2deg(data.left_ankle_rad),       'Chân trái ankle',       'L5', -90,  90, -60, -20;

    rad2deg(data.right_hip_yaw_rad),    'Chân phải hip yaw',     'R1', -30,  30, -0.5,  0.5;
    rad2deg(data.right_hip_roll_rad),   'Chân phải hip roll',    'R2', -25,  25, -30,  0;
    rad2deg(data.right_hip_pitch_rad),  'Chân phải hip pitch',   'R3', -30,  70, 20,  50;
    rad2deg(data.right_knee_rad),       'Chân phải knee',        'R4', -90,  90, 60, 100;
    rad2deg(data.right_ankle_rad),      'Chân phải ankle',       'R5', -90,  90, -60, -20
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

    % Kiểm tra khớp có gần hoặc vượt giới hạn hay không
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
        'FontSize', 12);

    ylabel('Amplitude (deg)', ...
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