%% Plot results from saved CSV files
% This script reads the simulation results from CSV files and creates
% visualizations without having to run the full simulation again

% Clear workspace
clear;
close all;
clc;

% Read the data from CSV files
results = readtable('simulation_results.csv');
summary = readtable('simulation_summary.csv');

% Extract data
rounds = results.Round;
operating_nodes = results.OperatingNodes;
dead_nodes = results.DeadNodes;
energy_consumed = results.EnergyConsumed;

% Plot 1: Operating Nodes per Round
figure(1);
plot(rounds, operating_nodes, '-b', 'LineWidth', 1.5);
title('Operating Nodes per Round');
xlabel('Rounds');
ylabel('Number of Operating Nodes');
grid on;
saveas(gcf, 'operating_nodes.png');

% Plot 2: Dead Nodes per Round
figure(2);
plot(rounds, dead_nodes, '-r', 'LineWidth', 1.5);
title('Dead Nodes per Round');
xlabel('Rounds');
ylabel('Number of Dead Nodes');
grid on;
saveas(gcf, 'dead_nodes.png');

% Plot 3: Energy Consumed per Round
figure(3);
plot(rounds, energy_consumed, '-g', 'LineWidth', 1.5);
title('Energy Consumed per Round');
xlabel('Rounds');
ylabel('Energy (J)');
grid on;
saveas(gcf, 'energy_consumed.png');

% Plot 4: Cumulative Energy Consumption
figure(4);
plot(rounds, cumsum(energy_consumed), '-m', 'LineWidth', 1.5);
title('Cumulative Energy Consumption');
xlabel('Rounds');
ylabel('Total Energy (J)');
grid on;
saveas(gcf, 'cumulative_energy.png');

% Display summary statistics
fprintf('\n-------- Simulation Summary --------\n');
fprintf('Total Nodes: %d\n', summary.TotalNodes);
fprintf('First node death: Round %d\n', summary.FirstNodeDeathRound);
fprintf('Half nodes death: Round %d\n', summary.HalfNodesDeathRound);
fprintf('All nodes death: Round %d\n', summary.AllNodesDeathRound);
fprintf('Total rounds: %d\n', max(rounds));

% You can add more plots as needed, like:
% - Network lifetime (time until first/half/all nodes die)
% - Comparison of different energy thresholds
% - Energy distribution among nodes
% etc.
