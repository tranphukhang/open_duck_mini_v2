clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'sagittal_motion.csv');

data = readtable(csv_file);

t = data.time_s;
x_right = data.right_foot_x_m;
x_left  = data.left_foot_x_m;
x_com   = data.com_x_m;
x_dcm   = data.dcm_x_m;

figure('Color', 'w');

plot(t, x_right, 'k-', 'LineWidth', 1.5);
hold on;
plot(t, x_left, 'r-', 'LineWidth', 1.5);
plot(t, x_com, 'b-', 'LineWidth', 1.8);
plot(t, x_dcm, 'm--', 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (m)', 'FontSize', 12);

title('Chuyển động theo mặt phẳng dọc', ...
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