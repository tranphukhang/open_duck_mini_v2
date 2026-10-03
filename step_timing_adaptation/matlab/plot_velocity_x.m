clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'matlab', 'data', 'velocity_x.csv');

data = readtable(csv_file);

t = data.time_s;
vx_des = data.desired_vx_m_s;
vx_actual = data.actual_com_vx_m_s;

figure('Color', 'w');

plot(t, vx_des, 'k-', 'LineWidth', 1.8);
hold on;
plot(t, vx_actual, 'r-', 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (m/s)', 'FontSize', 12);
title('Vận tốc CoM theo trục x', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('Vận tốc đặt', 'Vận tốc CoM', ...
    'Location', 'best', 'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t(1) t(end)]);