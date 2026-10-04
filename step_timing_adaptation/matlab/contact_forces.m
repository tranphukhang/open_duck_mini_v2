clc;
clear;
close all;

script_dir = fileparts(mfilename('fullpath'));

left_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'left_contact_force.csv');

right_file = fullfile(script_dir, '..', 'matlab', 'data', ...
    'right_contact_force.csv');

left_data = readtable(left_file);
right_data = readtable(right_file);

mu = 0.6;

%% LEFT FOOT DATA
t_left = left_data.time_s;

Fx_left = left_data.force_x_N;
Fy_left = left_data.force_y_N;
Fz_left = left_data.force_z_N;

Ft_left = sqrt(Fx_left.^2 + Fy_left.^2);
muFz_left = mu .* Fz_left;

%% RIGHT FOOT DATA
t_right = right_data.time_s;

Fx_right = right_data.force_x_N;
Fy_right = right_data.force_y_N;
Fz_right = right_data.force_z_N;

Ft_right = sqrt(Fx_right.^2 + Fy_right.^2);
muFz_right = mu .* Fz_right;

%% =========================================================
% LEFT FOOT - Fx
% ==========================================================
figure('Color', 'w');

plot(t_left, Fx_left, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân trái theo trục x', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_x', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_left(1) t_left(end)]);
ylim([-5 5]);

%% =========================================================
% LEFT FOOT - Fy
% ==========================================================
figure('Color', 'w');

plot(t_left, Fy_left, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân trái theo trục y', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_y', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_left(1) t_left(end)]);
ylim([-6 6]);

%% =========================================================
% LEFT FOOT - Fz
% ==========================================================
figure('Color', 'w');

plot(t_left, Fz_left, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân trái theo trục z', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_z', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_left(1) t_left(end)]);
ylim([-2 25]);

%% =========================================================
% RIGHT FOOT - Fx
% ==========================================================
figure('Color', 'w');

plot(t_right, Fx_right, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân phải theo trục x', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_x', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_right(1) t_right(end)]);
ylim([-5 5]);

%% =========================================================
% RIGHT FOOT - Fy
% ==========================================================
figure('Color', 'w');

plot(t_right, Fy_right, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân phải theo trục y', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_y', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_right(1) t_right(end)]);
ylim([-6 6]);

%% =========================================================
% RIGHT FOOT - Fz
% ==========================================================
figure('Color', 'w');

plot(t_right, Fz_right, 'r-', ...
    'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Lực tiếp xúc chân phải theo trục z', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend('F_z', ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_right(1) t_right(end)]);
ylim([-2 25]);

%% =========================================================
% LEFT FOOT - FRICTION CONE
% ==========================================================
figure('Color', 'w');

hLimitLeft = plot(t_left, muFz_left, 'k-', ...
    'LineWidth', 1.5);

hold on;

hFtLeft = plot(t_left, Ft_left, 'r--', ...
    'LineWidth', 1.8);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Điều kiện nón ma sát chân trái', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend([hFtLeft hLimitLeft], ...
    {'F_t', '\muF_z'}, ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_left(1) t_left(end)]);
ylim([0 15]);

%% =========================================================
% RIGHT FOOT - FRICTION CONE
% ==========================================================
figure('Color', 'w');

hLimitRight = plot(t_right, muFz_right, 'k-', ...
    'LineWidth', 1.5);

hold on;

hFtRight = plot(t_right, Ft_right, 'r--', ...
    'LineWidth', 1.8);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude (N)', 'FontSize', 12);

title('Điều kiện nón ma sát chân phải', ...
    'FontSize', 13, ...
    'FontName', 'Times New Roman', ...
    'FontWeight', 'normal');

legend([hFtRight hLimitRight], ...
    {'F_t', '\muF_z'}, ...
    'Location', 'best', ...
    'FontSize', 11);

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.0, ...
    'FontName', 'Times New Roman');

xlim([t_right(1) t_right(end)]);
ylim([0 15]);