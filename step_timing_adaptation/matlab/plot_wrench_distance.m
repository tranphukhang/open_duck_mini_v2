clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));

wrench_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'wrench_distance.csv');

data = readtable(wrench_file);

t = data.time_s;
d = data.distance;

%% ============================================================
% PLOT
% ============================================================

figure('Color', 'w');

plot(t, d, 'r-', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'd');

xlabel('Time (s)', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

ylabel('Amplitude', ...
    'FontSize', 24, ...
    'FontName', 'Times New Roman');

legend('Location', 'best', ...
    'Interpreter', 'latex', ...
    'FontSize', 16);

grid on;
box on;

set(gca, ...
    'FontSize', 22, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t(1) t(end)]);
ylim([0 6e-13]);