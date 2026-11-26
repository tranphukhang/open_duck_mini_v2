clc;
clear;
close all;

% ============================================================
% ZMP TRAJECTORY, SUPPORT POLYGON AND SIGNED STABILITY MARGIN
% ============================================================
% Input files from the same rl/log/eval_* directory:
%   states.csv          : time, ZMP and foot-contact states
%   support_polygon.csv : World XY vertices of the support hull
%
% Signed ZMP margin = distance to the closest boundary segment:
%   > 0 : ZMP inside the support polygon
%   = 0 : ZMP on the boundary
%   < 0 : ZMP outside the support polygon
%   NaN : no valid ZMP or no valid polygon
%
% No figures or processed data are saved automatically.

%% 1. SELECT EVALUATION DATA
script_dir = fileparts(mfilename('fullpath'));
log_dir = fullfile(script_dir, '..', 'log');
candidates = dir(fullfile(log_dir, 'eval_*', 'states.csv'));

if isempty(candidates)
    error('No states.csv was found in: %s', log_dir);
elseif numel(candidates) == 1
    states_file = fullfile(candidates(1).folder, candidates(1).name);
else
    [~, newest_idx] = max([candidates.datenum]);
    initial_file = fullfile(candidates(newest_idx).folder, ...
        candidates(newest_idx).name);

    [csv_name, csv_folder] = uigetfile( ...
        {'states.csv', 'MuJoCo state CSV (states.csv)'}, ...
        'Select evaluation states.csv', initial_file);

    if isequal(csv_name, 0)
        error('No evaluation data was selected.');
    end
    states_file = fullfile(csv_folder, csv_name);
end

polygon_file = fullfile(fileparts(states_file), 'support_polygon.csv');
if ~isfile(polygon_file)
    error('Matching support_polygon.csv not found: %s', polygon_file);
end

%% 2. IMPORT MATCHING CSV FILES
state_columns = { ...
    'sample_index', 'time_s', ...
    'contact_left', 'contact_right', ...
    'zmp_x_w_m', 'zmp_y_w_m'};

opts = detectImportOptions(states_file);
missing = setdiff(state_columns, opts.VariableNames);
if ~isempty(missing)
    error('Missing states.csv columns: %s', strjoin(missing, ', '));
end
opts.SelectedVariableNames = state_columns;
S = readtable(states_file, opts);

polygon_columns = { ...
    'sample_index', 'vertex_index', 'x_w_m', 'y_w_m'};

opts = detectImportOptions(polygon_file);
missing = setdiff(polygon_columns, opts.VariableNames);
if ~isempty(missing)
    error('Missing support_polygon.csv columns: %s', ...
        strjoin(missing, ', '));
end
opts.SelectedVariableNames = polygon_columns;
P = readtable(polygon_file, opts);

sample_id = double(S.sample_index);
t = double(S.time_s);
zmp_x = double(S.zmp_x_w_m);
zmp_y = double(S.zmp_y_w_m);
contact_left = double(S.contact_left);
contact_right = double(S.contact_right);

polygon_sample_id = double(P.sample_index);
polygon_vertex_id = double(P.vertex_index);
polygon_x = double(P.x_w_m);
polygon_y = double(P.y_w_m);

%% 3. PLOT SETTINGS
% Display 0-5 seconds; use [0, 10] for the full simulation.
time_limits = [0, 5];

% Show one historical support outline approximately every 0.10 seconds.
% Contact-mode changes are always included.
polygon_interval_s = 0.10;

% Leave [] for automatic Y-axis limits, or specify e.g. [-30, 30] (mm).
margin_y_limits_mm = [];

% Figure size (cm)
fig_width = 8;
fig_height = 8;

% Font settings
font_name = 'Times New Roman';
font_axis = 10;
font_label = 11;
font_legend = 8;

%% 4. RECONSTRUCT EVERY POLYGON AND CALCULATE THE ZMP MARGIN
n = numel(t);
polygons = cell(n, 1);
margin_m = nan(n, 1);

for k = 1:n
    % Match polygons to states by sample_index (NOT rounded time).
    row_idx = find(polygon_sample_id == sample_id(k));
    if numel(row_idx) < 3
        continue;
    end

    % Restore convex-hull vertex order from the CSV.
    [~, order] = sort(polygon_vertex_id(row_idx));
    row_idx = row_idx(order);
    V = [polygon_x(row_idx), polygon_y(row_idx)];

    if any(~isfinite(V(:)))
        continue;
    end

    % The vertices describe the OUTER support hull. Do not connect
    % successive polygons or individual feet inside the double-support hull.
    polygons{k} = V;

    if ~isfinite(zmp_x(k)) || ~isfinite(zmp_y(k))
        continue;
    end

    % Minimum Euclidean distance from ZMP to each finite EDGE SEGMENT.
    % Works for convex hulls with 4, 6, ... vertices.
    A = V;
    B = V([2:end, 1], :);
    AB = B - A;
    edge_len2 = sum(AB.^2, 2);
    valid_edges = edge_len2 > 1e-16;

    if ~any(valid_edges)
        continue;
    end

    A = A(valid_edges, :);
    AB = AB(valid_edges, :);
    edge_len2 = edge_len2(valid_edges);
    point = [zmp_x(k), zmp_y(k)];

    AP = point - A;
    u = sum(AP .* AB, 2) ./ edge_len2;
    u = min(1, max(0, u));

    nearest_points = A + u .* AB;
    distances = sqrt(sum((point - nearest_points).^2, 2));
    d_min = min(distances);

    % Assign sign using polygon membership.
    [inside, on_boundary] = inpolygon( ...
        point(1), point(2), V(:, 1), V(:, 2));

    if on_boundary
        margin_m(k) = 0;
    elseif inside
        margin_m(k) = d_min;
    else
        margin_m(k) = -d_min;
    end
end

margin_mm = 1000 * margin_m;

%% 5. FIGURE 1 - ZMP TRAJECTORY AND SUPPORT POLYGON (WORLD XY)
plot_mask = isfinite(t) & ...
    t >= time_limits(1) & t <= time_limits(2);
plot_ids = find(plot_mask);

if isempty(plot_ids)
    error('No data samples inside the selected time limits.');
end

last_polygon_idx = [];
for j = numel(plot_ids):-1:1
    k = plot_ids(j);
    if size(polygons{k}, 1) >= 3
        last_polygon_idx = k;
        break;
    end
end

fig1 = figure( ...
    'Name', 'ZMP and Support Polygon', ...
    'Color', 'w', 'Units', 'centimeters', ...
    'Position', [5, 5, fig_width, fig_height]);
set(fig1, 'PaperUnits', 'centimeters', ...
    'PaperSize', [fig_width, fig_height], ...
    'PaperPosition', [0, 0, fig_width, fig_height]);
hold on;

% Color historical hulls by simulation time, like rl/plotter.py.
colors = parula(256);
colormap(gca, colors);
caxis(time_limits);
last_draw_time = -inf;
last_contact_mode = [NaN, NaN];
h_history = gobjects(0);

for j = 1:numel(plot_ids)
    k = plot_ids(j);
    V = polygons{k};
    if size(V, 1) < 3 || k == last_polygon_idx
        continue;
    end

    mode = [contact_left(k), contact_right(k)];
    contact_change = ~isequaln(mode, last_contact_mode);
    if (t(k) - last_draw_time < polygon_interval_s - 1e-9) ...
            && ~contact_change
        continue;
    end

    normalized_time = (t(k) - time_limits(1)) / ...
        max(eps, diff(time_limits));
    c_idx = max(1, min(256, round(1 + 255 * normalized_time)));
    % Lighten the boundaries to keep the ZMP trajectory legible.
    color = 0.55 * [1, 1, 1] + 0.45 * colors(c_idx, :);
    V_closed = [V; V(1, :)];
    h = plot(V_closed(:, 1), V_closed(:, 2), '-', ...
        'Color', color, 'LineWidth', 0.65);
    if isempty(h_history)
        h_history = h;
    end
    last_draw_time = t(k);
    last_contact_mode = mode;
end

% The current (last valid) polygon is one CLOSED exterior outline.
h_current_polygon = gobjects(0);
if ~isempty(last_polygon_idx)
    V = polygons{last_polygon_idx};
    V_closed = [V; V(1, :)];
    h_current_polygon = plot(V_closed(:, 1), V_closed(:, 2), ...
        'b-', 'LineWidth', 1.6);
end

h_zmp = plot(zmp_x(plot_mask), zmp_y(plot_mask), ...
    'r-', 'LineWidth', 1.2);

% Mark the last valid ZMP within the selected time window.
valid_zmp_ids = find(plot_mask & isfinite(zmp_x) & isfinite(zmp_y));
h_zmp_current = gobjects(0);
if ~isempty(valid_zmp_ids)
    k = valid_zmp_ids(end);
    h_zmp_current = plot(zmp_x(k), zmp_y(k), 'ro', ...
        'MarkerFaceColor', 'r', 'MarkerSize', 4);
end

xlabel('World X (m)', 'FontName', font_name, 'FontSize', font_label);
ylabel('World Y (m)', 'FontName', font_name, 'FontSize', font_label);
axis equal;
grid on;
box on;
set(gca, 'FontName', font_name, 'FontSize', font_axis, ...
    'LineWidth', 0.8);

legend_handles = gobjects(0);
legend_names = {};
if ~isempty(h_history)
    legend_handles(end+1) = h_history;
    legend_names{end+1} = 'Support History';
end
if ~isempty(h_current_polygon)
    legend_handles(end+1) = h_current_polygon;
    legend_names{end+1} = 'Support Boundary';
end
legend_handles(end+1) = h_zmp;
legend_names{end+1} = 'ZMP';
if ~isempty(h_zmp_current)
    legend_handles(end+1) = h_zmp_current;
    legend_names{end+1} = 'Current ZMP';
end
legend(legend_handles, legend_names, 'Location', 'best', ...
    'FontName', font_name, 'FontSize', font_legend);

cb = colorbar;
cb.Label.String = 'Time (s)';
cb.FontName = font_name;
cb.FontSize = font_axis;
hold off;

%% 6. FIGURE 2 - SIGNED ZMP MARGIN VERSUS TIME
fig2 = figure( ...
    'Name', 'ZMP Stability Margin', ...
    'Color', 'w', 'Units', 'centimeters', ...
    'Position', [14, 5, fig_width, fig_height]);
set(fig2, 'PaperUnits', 'centimeters', ...
    'PaperSize', [fig_width, fig_height], ...
    'PaperPosition', [0, 0, fig_width, fig_height]);

plot(t(plot_mask), zeros(nnz(plot_mask), 1), ...
    'k--', 'LineWidth', 1.0);
hold on;
plot(t(plot_mask), margin_mm(plot_mask), ...
    'r-', 'LineWidth', 1.2);
hold off;

xlabel('Time (s)', 'FontName', font_name, 'FontSize', font_label);
ylabel('ZMP Margin (mm)', 'FontName', font_name, 'FontSize', font_label);
legend('Boundary', 'ZMP Margin', 'Location', 'best', ...
    'FontName', font_name, 'FontSize', font_legend);
grid on;
box on;
set(gca, 'FontName', font_name, 'FontSize', font_axis, ...
    'LineWidth', 0.8);
xlim(time_limits);
if ~isempty(margin_y_limits_mm)
    ylim(margin_y_limits_mm);
end

%% 7. PRINT SUMMARY
valid_margin = isfinite(margin_mm);
fprintf('States:   %s\n', states_file);
fprintf('Polygons: %s\n', polygon_file);
fprintf('Total samples: %d\n', n);
fprintf('Valid ZMP margin samples: %d\n', nnz(valid_margin));
if any(valid_margin)
    fprintf('Minimum signed margin: %.3f mm\n', min(margin_mm(valid_margin)));
    fprintf('Maximum signed margin: %.3f mm\n', max(margin_mm(valid_margin)));
    fprintf('ZMP outside support: %d samples\n', nnz(margin_mm < 0));
end
