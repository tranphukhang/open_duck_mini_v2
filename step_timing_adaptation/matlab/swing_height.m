clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));

left_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'left_swing_height.csv');

right_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'right_swing_height.csv');

left_data = readtable(left_file);
right_data = readtable(right_file);

t_left = left_data.time_s;
z_left = left_data.left_foot_z_m;

t_right = right_data.time_s;
z_right = right_data.right_foot_z_m;

%% LEFT SWING HEIGHT
figure('Color', 'w');

plot(t_left, z_left, 'r-', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Chân trái');

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (m)', 'FontSize', 12);

title('Chiều cao chân trái', ...
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

xlim([t_left(1) t_left(end)]);

%% RIGHT SWING HEIGHT
figure('Color', 'w');

plot(t_right, z_right, 'r-', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Chân phải');

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (m)', 'FontSize', 12);

title('Chiều cao chân phải', ...
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

xlim([t_right(1) t_right(end)]);