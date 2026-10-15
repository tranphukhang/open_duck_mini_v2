clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'velocity_y.csv');

data = readtable(csv_file);

t = data.time_s;
vy_des = data.desired_vy_m_s;
vy_actual = data.actual_com_vy_m_s;

figure('Color', 'w');

plot(t, vy_des, 'k-', 'LineWidth', 1.8);
hold on;
plot(t, vy_actual, 'r-', 'LineWidth', 1.5);

xlabel('Time (s)', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

ylabel('Amplitude (m/s)', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

legend('Desired velocity', 'CoM velocity', ...
    'Location', 'best', ...
    'FontSize', 16, ...
    'FontName', 'Times New Roman');

grid on;
box on;

set(gca, ...
    'FontSize', 22, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t(1) t(end)]);
ylim([-0.5 0.7]);