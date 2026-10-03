clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));

wrench_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'wrench_distance.csv');

left_force_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'left_contact_force.csv');

right_force_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'right_contact_force.csv');

wrench_data = readtable(wrench_file);
left_data = readtable(left_force_file);
right_data = readtable(right_force_file);

t = wrench_data.time_s;
d = wrench_data.distance;

F_left = sqrt( ...
    left_data.force_x_N.^2 + ...
    left_data.force_y_N.^2 + ...
    left_data.force_z_N.^2);

F_right = sqrt( ...
    right_data.force_x_N.^2 + ...
    right_data.force_y_N.^2 + ...
    right_data.force_z_N.^2);

contact_tol = 1e-6;

left_contact = F_left > contact_tol;
right_contact = F_right > contact_tol;

SS = xor(left_contact, right_contact);
DS = left_contact & right_contact;

d_SS = d;
d_DS = d;

d_SS(~SS) = NaN;
d_DS(~DS) = NaN;

figure('Color', 'w');

plot(t, d_SS, 'r-', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'SS');

hold on;

plot(t, d_DS, 'b-', ...
    'LineWidth', 1.5, ...
    'DisplayName', 'DS');

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);

title('Khoảng cách wrench', ...
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
ylim([0 4e-13]);