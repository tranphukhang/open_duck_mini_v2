clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'lateral_motion.csv');

data = readtable(csv_file);

t = data.time_s;
y_right = data.right_foot_y_m;
y_left  = data.left_foot_y_m;
y_com   = data.com_y_m;
y_dcm   = data.dcm_y_m;

figure('Color', 'w');

plot(t, y_right, 'k-', 'LineWidth', 1.5);
hold on;
plot(t, y_left, 'r-', 'LineWidth', 1.5);
plot(t, y_com, 'b-', 'LineWidth', 1.8);
plot(t, y_dcm, 'm--', 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (m)', 'FontSize', 12);

title('Chuyển động theo mặt phẳng ngang', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('Chân phải', ...
       'Chân trái', ...
       'CoM', ...
       'DCM', ...
       'Location', 'best', ...
       'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t(1) t(end)]);
ylim([-0.04 0.12]);