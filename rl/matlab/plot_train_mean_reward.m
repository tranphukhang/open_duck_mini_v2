%% PPO Training - Mean Reward
% CSV exported from TensorBoard: Train/mean_reward
% CSV columns: Wall time, Step, Value

clearvars;
clc;
close all;

%% 1. Load data
script_dir = fileparts(mfilename('fullpath'));
csv_file = fullfile(script_dir, '..', 'log', ...
    '2026-11-19_23-14-31_flat_locomotion.csv');

if ~isfile(csv_file)
    error('Cannot find TensorBoard CSV file:\n%s', csv_file);
end

T = readtable(csv_file, 'VariableNamingRule', 'preserve');

if ~all(ismember({'Step', 'Value'}, T.Properties.VariableNames))
    error('CSV file must contain Step and Value columns.');
end

iteration = double(T.('Step'));
mean_reward = double(T.('Value'));

valid = isfinite(iteration) & isfinite(mean_reward);
iteration = iteration(valid);
mean_reward = mean_reward(valid);

if isempty(iteration)
    error('No valid PPO reward samples found in the CSV file.');
end

%% 2. Plot training reward
figure('Name', 'PPO Training Mean Reward', ...
       'Color', 'w', 'Position', [150, 150, 950, 520]);

plot(iteration, mean_reward, 'r-', 'LineWidth', 1.5);

xlabel('PPO Training Iteration', 'FontSize', 13);
ylabel('Mean Reward', 'FontSize', 13);
legend('Mean Reward', 'Location', 'best');

grid on;
box on;
set(gca, 'FontName', 'Times New Roman', ...
         'FontSize', 12, 'LineWidth', 1.0);
xlim([min(iteration), max(iteration)]);

%% 3. Summary
fprintf('Loaded %d valid reward samples.\n', numel(iteration));
fprintf('PPO training iterations: %.0f to %.0f.\n', ...
        min(iteration), max(iteration));
fprintf('Last recorded mean reward: %.4f.\n', mean_reward(end));
