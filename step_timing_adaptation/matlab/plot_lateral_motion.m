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
plot(t, y_com, 'b-', 'LineWidth', 1.5);
plot(t, y_dcm, 'm--', 'LineWidth', 1.5);

xlabel('Time (s)', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

ylabel('Amplitude (m)', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

legend('Right foot', ...
       'Left foot', ...
       'CoM', ...
       'DCM', ...
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
ylim([-0.07 0.2]);