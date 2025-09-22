%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       EER-RL EPOCH RUNNER                            %
% This script runs multiple epochs of a WSN simulation based on EER-RL  %
% and tracks first node death rounds                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;
close all;

% Configuration
total_epochs = 300;      % Number of epochs to run for full analysis
results_file = 'first_node_death_epochs_30nodes.csv';  % Output file name (for 30 nodes)
save_interval = 10;       % Save results every X epochs

% Array to store results
first_death_rounds = zeros(total_epochs, 1);

% Create or overwrite results file with header
fileID = fopen(results_file, 'w');
fprintf(fileID, 'Epoch,FirstNodeDeathRound\n');
fclose(fileID);

fprintf('Starting simulation for %d epochs...\n', total_epochs);

% Run each epoch
for epoch_num = 1:total_epochs
    % Set random seed for reproducibility
    rng(epoch_num);
    
    % Setup WSN simulation parameters
    xm=100;
    ym=100;
    x=0;
    y=0;
    n=30;
    dead_nodes=0;
    sinkx=50;
    sinky=50;
    send_energy = 0.3;
    receive_energy = 0.2; 
    active_energy = 0.1;
    k=4000;
    rnd=0;
    tot_rnd=3000;
    op_nodes=n;
    transmissions=0;
    d(n,n)=0;
    source=1;
    flag1stdead=0;
    range_C = 20;
    alpha=1;
    gamma = 0.95;
    p=0.5;
    q1=1-p;
    CH_tot= ceil(n*0.1);
    first_node_death_round = -1;
    
    % Create WSN
    for i=1:n
        NET(i).id=i;
        NET(i).x=rand(1,1)*xm;
        NET(i).y=rand(1,1)*ym;
        NET(i).E = 150 + (rand(1,1)*20 - 10);
        NET(i).Eo = NET(i).E;
        NET(i).cond=1;
        NET(i).dts= sqrt((sinkx-NET(i).x)^2 + (sinky-NET(i).y)^2);
        NET(i).hop=ceil(NET(i).dts/range_C);
        NET(i).role=0;
        NET(i).closest=0;
        NET(i).prev=0;
        NET(i).sel=0;
        NET(i).rop=0;
        NET(i).dest=0;
        NET(i).dts_ch=0;
    end
    
    % Compute Q-Value
    min_E = min([NET.E]); 
    max_E = max([NET.E]);
    for i=1:n
        if(min_E ==max_E)
            Q(i) = 1 / NET(i).hop;
            NET(i).Q = Q(i);
        else
            Q(i) = (p*(NET(i).E - min_E)/(max_E-min_E)+(q1/NET(i).hop));
            NET(i).Q = Q(i);
        end
    end
    
    % Cluster Head Election
    for i=1:n
       CM(i) = NET(i);
    end
    tot = 1;
    
    while(tot<=CH_tot)
        for i=1:n
            if(CM(i).Q == max([CM.Q]))
                if tot == 1 && CM(i).dts>=15 && CM(i).dts<=50
                    CH(tot) = CM(i);
                    NET(i).role=1;
                    CM(i).Q = 0;
                    tot = tot+1;
                elseif tot>1 &&  CM(i).dts>=15 && CM(i).dts<=50
                    cl = 0;
                    for t = 1:length(CH)
                        dts = sqrt((CM(i).x-CH(t).x)^2 + (CM(i).y-CH(t).y)^2);
                        if(dts <=15)
                            cl=cl +1;
                            break;
                        end
                    end
                    if cl==0
                        CH(tot) = CM(i);
                        NET(i).role=1;
                        CM(i).Q = 0;
                        tot =tot+1;  
                    else
                        CM(i).Q = 0;
                    end
                else
                    CM(i).Q = 0;
                end
            end
            if tot >CH_tot
               break;
           end
        end
        
        % If all Q values are 0 or negative, break to avoid infinite loop
        if max([CM.Q]) <= 0
            break;
        end
    end
    
    % Cluster Formation
    for i=1:n
        for ch=1:length(CH)
            dts_ch(i,ch) = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
            dts_ch(dts_ch==0)=NaN;
        end
        if NET(i).dts<=range_C
            NET(i).dest = 0;
            NET(i).dts_ch = 0;
        else
            if NET(i).role==0
                for ch = 1:length(CH)
                    dtsCh = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
                    if NET(i).E>0 && dtsCh <= min([dts_ch(i,:)])
                        NET(i).dest = CH(ch).id;
                        NET(i).dts_ch = min([dts_ch(i,:)]);
                        break;
                    end
                end
                % Even if no CH is in range, assign to the closest CH for energy calculation
                if NET(i).dest == 0 && NET(i).E > 0 && NET(i).role == 0
                    [minDist, minIdx] = min([dts_ch(i,:)]);
                    if ~isempty(minIdx) && ~isnan(minDist)
                        NET(i).dest = CH(minIdx).id;
                        NET(i).dts_ch = minDist;
                    end
                end
            end  
        end             
    end
    
    % Main simulation loop
    while(op_nodes>0 && rnd<tot_rnd && first_node_death_round == -1)
        % Increment round counter
        rnd = rnd + 1;
        
        % Energy consumption simulation
        for i=1:n
            if NET(i).E > 0
                % Different energy consumption based on role
                if NET(i).role == 1
                    % Cluster head - higher consumption
                    NET(i).E = NET(i).E - send_energy - active_energy*0.8;
                else
                    % Regular node
                    if NET(i).dts <= range_C
                        % Direct to sink
                        NET(i).E = NET(i).E - send_energy*0.8 - active_energy*0.5;
                    else
                        % To cluster head
                        NET(i).E = NET(i).E - send_energy*0.7 - active_energy*0.4;
                    end
                end
                
                % Ensure non-negative energy
                if NET(i).E < 0
                    NET(i).E = 0;
                    NET(i).cond = 0;
                end
            end
        end
        
        % Check for dead nodes
        dead = 0;
        for i=1:n
            if NET(i).E <= 0 || NET(i).cond == 0
                dead = dead + 1;
                NET(i).cond = 0;
            end
        end
        
        % Update operational nodes
        op_nodes = n - dead;
        
        % Track first node death
        if dead > 0 && first_node_death_round == -1
            first_node_death_round = rnd;
        end
    end
    
    % If simulation ended without any node dying, set to max rounds
    if first_node_death_round == -1
        first_node_death_round = rnd;
    end
    
    % Record result
    first_death_rounds(epoch_num) = first_node_death_round;
    
    % Show progress
    fprintf('Epoch %d/%d (30 nodes): First node death at round %d\n', epoch_num, total_epochs, first_node_death_round);
    
    % Save results periodically
    if mod(epoch_num, save_interval) == 0 || epoch_num == total_epochs
        fileID = fopen(results_file, 'a');
        for i = max(1, epoch_num-save_interval+1):epoch_num
            fprintf(fileID, '%d,%d\n', i, first_death_rounds(i));
        end
        fclose(fileID);
    end
    
    % Clear variables for next epoch
    clear NET CM CH dts_ch Q;
end

% Calculate statistics
avg_first_death = mean(first_death_rounds);
min_first_death = min(first_death_rounds);
max_first_death = max(first_death_rounds);
std_first_death = std(first_death_rounds);

fprintf('\n========== FINAL RESULTS ==========\n');
fprintf('Total Epochs: %d\n', total_epochs);
fprintf('Average First Node Death Round: %.2f\n', avg_first_death);
fprintf('Minimum First Node Death Round: %d\n', min_first_death);
fprintf('Maximum First Node Death Round: %d\n', max_first_death);
fprintf('Standard Deviation: %.2f\n', std_first_death);
fprintf('Results saved to: %s\n', results_file);

% Create a simple histogram of first node death rounds
figure;
histogram(first_death_rounds, min(20, total_epochs));
title('Distribution of First Node Death Rounds (30 Nodes)');
xlabel('Round Number');
ylabel('Frequency');
saveas(gcf, 'first_node_death_histogram_30nodes.png');
