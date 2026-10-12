clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));

wrench_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'wrench_distance.csv');

wrench_data = readtable(wrench_file);

t = wrench_data.time_s;
d = wrench_data.distance;

figure('Color', 'w');

plot(t, d, 'k-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', ...
    'FontSize', 12, ...
    'FontName', 'Times New Roman');

ylabel('Amplitude', ...
    'FontSize', 12, ...
    'FontName', 'Times New Roman');

title('Wrench distance', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t(1) t(end)]);
ylim([0 1e-6]);