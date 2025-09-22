%% Compare results from different simulations
% This script compares results from different node counts or methods
% and creates side-by-side comparison plots

% Clear workspace
clear;
close all;
clc;

%% Define configurations to compare
% You can modify these arrays to include different node counts or methods
node_counts = [10, 30, 50, 100];
methods = {'EER_RL', 'Competitor'};  % Add other methods as needed

%% Compare network lifetime across configurations
figure('Position', [100, 100, 1000, 600]);

% First node death comparison
subplot(1,3,1);
first_node_death = zeros(length(methods), length(node_counts));
for m = 1:length(methods)
    for n = 1:length(node_counts)
        try
            % Try to load the summary data
            filename = sprintf('../%s/%d/%s_simulation_summary.csv', methods{m}, node_counts(n), lower(methods{m}));
            summary = readtable(filename);
            first_node_death(m,n) = summary.FirstNodeDeathRound;
        catch
            % If file doesn't exist, set to NaN
            first_node_death(m,n) = NaN;
            warning('Could not load file: %s', filename);
        end
    end
end

% Plot first node death comparison
bar(first_node_death');
title('First Node Death Comparison');
xlabel('Number of Nodes');
xticklabels(node_counts);
ylabel('Round');
legend(methods, 'Location', 'best');
grid on;

% Half nodes death comparison
subplot(1,3,2);
half_nodes_death = zeros(length(methods), length(node_counts));
for m = 1:length(methods)
    for n = 1:length(node_counts)
        try
            % Try to load the summary data
            filename = sprintf('../%s/%d/%s_simulation_summary.csv', methods{m}, node_counts(n), lower(methods{m}));
            summary = readtable(filename);
            half_nodes_death(m,n) = summary.HalfNodesDeathRound;
        catch
            % If file doesn't exist, set to NaN
            half_nodes_death(m,n) = NaN;
        end
    end
end

% Plot half nodes death comparison
bar(half_nodes_death');
title('Half Nodes Death Comparison');
xlabel('Number of Nodes');
xticklabels(node_counts);
ylabel('Round');
legend(methods, 'Location', 'best');
grid on;

% All nodes death comparison (network lifetime)
subplot(1,3,3);
all_nodes_death = zeros(length(methods), length(node_counts));
for m = 1:length(methods)
    for n = 1:length(node_counts)
        try
            % Try to load the summary data
            filename = sprintf('../%s/%d/%s_simulation_summary.csv', methods{m}, node_counts(n), lower(methods{m}));
            summary = readtable(filename);
            all_nodes_death(m,n) = summary.AllNodesDeathRound;
        catch
            % If file doesn't exist, set to NaN
            all_nodes_death(m,n) = NaN;
        end
    end
end

% Plot all nodes death comparison
bar(all_nodes_death');
title('Network Lifetime Comparison');
xlabel('Number of Nodes');
xticklabels(node_counts);
ylabel('Round');
legend(methods, 'Location', 'best');
grid on;

% Save the figure
saveas(gcf, '../All figures/network_lifetime_comparison.png');

%% Create detailed comparison for each node count
for n = 1:length(node_counts)
    figure('Position', [100, 100, 1000, 400]);
    node_count = node_counts(n);
    
    % Load data for each method
    data = cell(length(methods), 1);
    for m = 1:length(methods)
        try
            filename = sprintf('../%s/%d/%s_simulation_data.csv', methods{m}, node_count, lower(methods{m}));
            data{m} = readtable(filename);
        catch
            warning('Could not load file: %s', filename);
            data{m} = [];
        end
    end
    
    % Plot operating nodes over time
    if ~isempty(data{1}) && ~isempty(data{2})
        % Plot operating nodes comparison
        subplot(1,2,1);
        hold on;
        max_rounds = 0;
        for m = 1:length(methods)
            if ~isempty(data{m})
                plot(data{m}.Round, data{m}.OperatingNodes, 'LineWidth', 1.5);
                max_rounds = max(max_rounds, max(data{m}.Round));
            end
        end
        title(sprintf('Operating Nodes Comparison (%d nodes)', node_count));
        xlabel('Rounds');
        ylabel('Number of Operating Nodes');
        legend(methods, 'Location', 'best');
        xlim([0, max_rounds]);
        grid on;
        hold off;
        
        % Plot energy consumption comparison
        subplot(1,2,2);
        hold on;
        for m = 1:length(methods)
            if ~isempty(data{m})
                plot(data{m}.Round, cumsum(data{m}.EnergyConsumed), 'LineWidth', 1.5);
            end
        end
        title(sprintf('Cumulative Energy Consumption (%d nodes)', node_count));
        xlabel('Rounds');
        ylabel('Total Energy (J)');
        legend(methods, 'Location', 'best');
        xlim([0, max_rounds]);
        grid on;
        hold off;
        
        % Save the comparison figure
        saveas(gcf, sprintf('../All figures/overlay_comparison_%dnodes.png', node_count));
    end
end
