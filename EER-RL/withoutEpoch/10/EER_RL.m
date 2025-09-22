%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              EER-RL                                  %
%       Energy-Efficient Routing based on Reinforcement Learning       %        %                      %
%                      Mobile Information Systems                      %
%                           Research Article                           %
%                                                                      %
% (c) Vially KAZADI MUTOMBO, PhD Candidate                             %
% Soongsil University                                                  %
% Department of Computer Science                                       %
% mutombo.kazadi@gmail.com                                             %
% February 2021                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

%%%%%%%%%%%%%%%%%%%%%% Beginning  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Network Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sensing Field Dimensions in meters %
xm=60;
ym=60;
x=0; % added for better display results of the plot
y=0; % added for better display results of the plot
% Number of Nodes in the field %
n=10;
% Number of Dead Nodes in the beggining %
dead_nodes=0;
% Coordinates of the Sink (location is predetermined in this simulation) %
sinkx=30;
sinky=30;

%%% Energy parameters %%%
% Fixed energy values for transmission, reception, and active state %
send_energy = 0.3; % units in Joules (fixed value for sending data)
receive_energy = 0.2; % units in Joules (fixed value for receiving data)
active_energy = 0.1; % units in Joules (energy consumed per round for being active)
% Size of data package %
k=4000; % units in bits 
% Round of Operation %
rnd=0;
tot_rnd=3000;
% Current Number of operating Nodes %
op_nodes=n; %Operating nodes
transmissions=0;
d(n,n)=0;
source=1;
flag1stdead=0;
range_C = 10; %Transmission range
alpha=1; %Learning Rate
gamma = 0.95; % Discount Factor
p=0.5; % Energy's Probabilistic parameter 
q1=1-p; % Hop count probabilistic parameter

CH_tot= ceil(n*0.1);
%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Network settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% WSN Creattiom %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the WSN %
for i=1:n
    
    
    NET(i).id=i;	% sensor's ID number
    NET(i).x=rand(1,1)*xm;	% X-axis coordinates of sensor node
    NET(i).y=rand(1,1)*ym;	% Y-axis coordinates of sensor node
    %NET(i).E=Eo;     % nodes energy levels (initially set to be equal to "Eo
    NET(i).E = 100 + (rand(1,1)*20 - 10); % Initial energy: random between 90 and 110
    NET(i).Eo = NET(i).E;
    NET(i).cond=1;   % States the current condition of the node. when the node is operational its value is =1 and when dead =0
    %NET(i).dts=0;    % nodes distance from the sink
    NET(i).dts= sqrt((sinkx-NET(i).x)^2 + (sinky-NET(i).y)^2);
    NET(i).hop=ceil(NET(i).dts/range_C); %Hop count estimate to the sink
    NET(i).role=0;   % node acts as normal if the value is '0', if elected as a cluster head it  gets the value '1' (initially all nodes are normal)
    %NET(i).pos=0;
    %NET(i).first=0;  %Initial route available. If it the first time a node send a packet its value is 0, otherwise it's 1
    NET(i).closest=0;
    NET(i).prev=0;
    %NET(i).next=0;
    %NET(i).dis=0;	% distance between two nodes headin towards to the cluster head from position 1
    NET(i).sel=0;    % states if the node has already operated for this round or not (if 0 then no, if 1 then yes) 
    NET(i).rop=0;    % number of rounds node was operational
    NET(i).dest=0;
    NET(i).dts_ch=0;
    %order(i)=0;

    hold on;
    figure(1);
    plot(x,y,xm,ym,NET(i).x,NET(i).y,'ob','DisplayName','cm');
    plot(x,y,xm,ym,sinkx,sinky,'*r','DisplayName','sink');
    title 'EER-RL';
    xlabel '(m)';
    ylabel '(m)';
end

% find Neighbord nodes
%Compute Q-Value
min_E = min([NET.E]); 
max_E = max([NET.E]);
for i=1:n
    if(min_E ==max_E)
        Q(i) = 1 / NET(i).hop;
        NET(i).Q = Q(i);
    else
        Q(i) = (p*(NET(i).E - min_E)/(max_E-min_E)+(q1/NET(i).hop));
        NET(i).Q = Q(i);
        %CH = maxk(Q,10); %Find 10 strongest nodes 
    end
end
%------------------- BEGINNING OF CLUSTERING -----------------------------
%CLUSTER HEAD ELECTION
for i=1:n
   CM(i) = NET(i); %Make a copy of the network
end
tot = 1;

while(tot<=CH_tot)
for i=1:n
    %maxx= max([CM.Q]);
    %disp(maxx);
    if(CM(i).Q == max([CM.Q]))
        if tot == 1 && CM(i).dts>=15 && CM(i).dts<=50
            CH(tot) = CM(i);
            NET(i).role=1;
            
            plot(x,y,xm,ym,NET(i).x,NET(i).y,'Or','DisplayName','CH');
            CM(i).Q = 0;
            tot =tot+1;
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
                
                plot(x,y,xm,ym,NET(i).x,NET(i).y,'Or');
                
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
end
%END CLUSTER HEAD ELECTION

%CLUSTER FORMATION
for i=1:n
    for ch=1:CH_tot
        dts_ch(i,ch) = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
        dts_ch(dts_ch==0)=NaN;
    end
    if NET(i).dts<=range_C
        NET(i).dest = 0;
        NET(i).dts_ch = 0;
    else
        if NET(i).role==0
            for ch = 1:CH_tot
                dtsCh = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
                if NET(i).E>0 && dtsCh <= min([dts_ch(i,:)])
                    NET(i).dest = CH(ch).id;
                    NET(i).dts_ch = min([dts_ch(i,:)]);
                    
                    figure(1);
                    plot([NET(i).x CH(ch).x], [NET(i).y CH(ch).y],'-c','DisplayName','Tx');
                    %legend;
                    %hold off;
                    
                    break;
                end
            end
            % Even if no CH is in range, assign to the closest CH for energy calculation
            % This represents a failed transmission attempt
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
%END CLUSTER FORMATION
for i=1:CH_tot
    ncm=1;
    for j=1:n
        if(NET(j).dest == CH(i).id & NET(j).role==0)
            cluster(i,ncm)= NET(j);
            ncm= ncm+1;
        end
    end
end
%------------------- END OF CLUSTERING -----------------------------------



%COMMUNICATION PHASE------------------------------------------------------
dead=0;
zero_energy_rounds = 0; % Track consecutive rounds with zero energy consumption
while((op_nodes>0 && rnd<tot_rnd) && zero_energy_rounds < 5)
    
    %Node nearby the Sink node
    Next_sink = [];  % Initialize Next_sink array
    ns=1;
    for l=1:n
        if NET(l).E>0
            if NET(l).dts<=range_C && NET(l).role==0
                Next_sink(ns) = NET(l);
                ns = ns + 1;
            end
        end    
    end
   
    for j = 1:CH_tot
        for ns=1:length(Next_sink)
            dts_tmp(j,ns) = sqrt((CH(j).x-Next_sink(ns).x)^2 + (CH(j).y-Next_sink(ns).y)^2);               
        end
    end
     %en Node nearby the sink
    energy=0;
    %INTRACLUSTER MULTIHOP COMMUNICATION
    for i=1:CH_tot
        ncm=1;
        for j =1:length(cluster(i,:))        
            if cluster(i,j).dest == CH(i).id
                if cluster(i,j).E>0
                    maxQ = max([cluster(i,:).Q]);
                    if (cluster(i,j).dts_ch<=range_C | cluster(i,j).Q ==maxQ) 
                        if cluster(i,j).prev==0
                            ETx= send_energy;
                            cluster(i,j).E = cluster(i,j).E-ETx-active_energy;
                            NET(cluster(i,j).id).E=cluster(i,j).E;
                            energy=energy+ETx+active_energy;
                            CH(i).prev = CH(i).prev +1;
                        else
                            ERx=receive_energy;
                            ETx= send_energy;
                            NET(cluster(i,j).id).E=NET(cluster(i,j).id).E-ETx-ERx-active_energy;
                            cluster(i,j).E = NET(cluster(i,j).id).E;
                            cluster(i,j).prev=0;
                            energy=energy+ETx+ERx+active_energy;
                            CH(i).prev = CH(i).prev +1;
                        end
                        %Compute the reward
                        Q_old = cluster(i,j).Q;
                        R= (p*(cluster(i,j).E - min_E)/(max_E-min_E)+(q1/cluster(i,j).hop));
                        
                        %update Q_value
                        cluster(i,j).Q =Q_old + alpha*(R+ gamma * maxQ -Q_old) ;
                        NET(cluster(i,j).id).Q = cluster(i,j).Q;
                    else
                        % Node is out of transmission range but still attempts to send data
                        % First try to send through a node with max Q value
                        for nex = 1:length(cluster(i,:))
                            if(cluster(i,nex).E>0)
                                if(cluster(i,nex).Q ==maxQ)
                                    next = cluster(i,nex);
                                    cluster(i,nex).prev=1;
                                    nextID=nex;
                                    break;
                                end
                            else
                                cluster(i,nex).Q = -100;
                            end
                           
                        end
                        dts_cm = sqrt((next.x-cluster(i,j).x)^2 + (next.y-cluster(i,j).y)^2);   
                        if cluster(i,j).prev==0
                            ETx= send_energy;
                            NET(cluster(i,j).id).E=NET(cluster(i,j).id).E-ETx-active_energy;
                            cluster(i,j).E = cluster(i,j).E-ETx-active_energy;
                            energy=energy+ETx+active_energy;
                        else
                            ERx=receive_energy;
                            ETx= send_energy;
                            NET(cluster(i,j).id).E=NET(cluster(i,j).id).E-ETx-ERx-active_energy;
                            cluster(i,j).E = cluster(i,j).E-ETx-ERx-active_energy;
                            cluster(i,j).prev=0;
                            energy=energy+ETx+active_energy;
                        end
                        %Compute the reward
                        Q_old = cluster(i,j).Q;
                        R= (p*(cluster(i,j).E - min_E)/(max_E-min_E)+(q1/cluster(i,j).hop));
                        
                        %update Q_value
                        cluster(i,j).Q =Q_old + alpha*(R+ gamma * maxQ -Q_old) ;
                        NET(cluster(i,j).id).Q = cluster(i,j).Q;
                        
                        Q_old = NET(next.id).Q;
                        cluster(i,nextID).Q  =Q_old + alpha*(R+ gamma * maxQ -Q_old) ;
                        NET(cluster(i,nextID).id).Q = cluster(i,nextID).Q;
                    end
                
                else
                    cluster(i,j).Q = -100;
                end
                
                % For nodes that are completely out of transmission range of any cluster head
                % Still consume energy for attempted transmission
                if cluster(i,j).dest == CH(i).id && cluster(i,j).dts_ch > range_C && ~(cluster(i,j).Q == maxQ) && cluster(i,j).E > 0
                    % Node is completely out of range but attempts to transmit anyway
                    ETx = send_energy;
                    cluster(i,j).E = cluster(i,j).E - ETx - active_energy;
                    NET(cluster(i,j).id).E = cluster(i,j).E;
                    energy = energy + ETx + active_energy;
                    % Update Q value negatively due to failed transmission
                    cluster(i,j).Q = cluster(i,j).Q * 0.95;  % Decrease Q value
                    NET(cluster(i,j).id).Q = cluster(i,j).Q;
                end
                    
            end
        end
    end

    %END OF INTRACLUSTER MULTIHOP COMMUNICATION

    %INTERCLUSTER COMMUNICATION
    for j =1:CH_tot
        thres = NET(CH(j).id).Eo * 0.4;
        if CH(j).E >thres && thres>0
            if(CH(j).dts<=range_C)
                if CH(j).prev ==0
                    ETx= send_energy;
                    NET(CH(j).id).E=NET(CH(j).id).E-ETx-active_energy;
                    CH(j).E = CH(j).E-ETx-active_energy;

                    energy=energy+ETx+active_energy;
                else
                    %ERx=receive_energy;
                    %ETx= send_energy;
                    %Edis = (send_energy + receive_energy + active_energy);
                    Edis = (send_energy + receive_energy + active_energy);
                    NET(CH(j).id).E=NET(CH(j).id).E-Edis;
                    CH(j).E = CH(j).E-Edis;
                    energy=energy+Edis;
                    CH(j).prev =0;
                end
                
            else
                for ns=1:length(Next_sink)
                    if dts_tmp(j,ns) == min(dts_tmp(j,:))
                        if CH(j).prev ==0
                            ETx= send_energy;
                            NET(CH(j).id).E=NET(CH(j).id).E-ETx-active_energy;
                            CH(j).E = CH(j).E-ETx-active_energy;

                            energy=energy+ETx+active_energy;
                        else
                            %ERx=receive_energy;
                            %ETx= send_energy;
                            %Edis = (send_energy + receive_energy + active_energy);
                            Edis = (send_energy + receive_energy + active_energy);
                            NET(CH(j).id).E=NET(CH(j).id).E-Edis;
                            CH(j).E = CH(j).E-Edis;
                            energy=energy+Edis;
                            CH(j).prev =0;
                        end  
                        NET(Next_sink(ns).id).prev = 1;
                        break;
                    end
                    
                end
            end
            
            %Compute the reward
            Q_old = CH(j).Q;
            R= (p*(CH(j).E - min_E)/(max_E-min_E)+(q1/CH(j).hop));
                        
            %update Q_value
            CH(j).Q =Q_old + alpha*(R+ gamma * maxQ -Q_old) ;
            NET(CH(j).id).Q = CH(j).Q;
        elseif CH(j).E <= thres || thres<=0
            
            %------------------- BEGINNING OF RECLUSTERING --------------
            %CLUSTER HEAD ELECTION
            aln =0; %Alive nodes before reclustering
            for i=1:n
                NET(i).dest =0;
                NET(i).dts_ch =0;
                NET(i).role=0;
                if NET(i).E>0
                    NET(i).Eo = NET(i).E;
                    CM(i) = NET(i);
                    aln = aln+1; 
                else
                    NET(i).Eo = NET(i).E;
                    NET(i).cond = 0;
                    
                    
                end    
            end
            CH_tot = ceil(aln/10);
            if CH_tot < 1
                CH_tot = 1; % Ensure at least 1 cluster head
            end
            %disp("NA ="+CH_tot+" ALN="+aln+" and N="+n)
            tot = 1;
            attempts = 0;
            max_attempts = n * 2; % Prevent infinite loops
            
            while(tot<=CH_tot && attempts < max_attempts)
                attempts = attempts + 1;
                max_Q = max([CM.Q]);
                
                % If all Q values are 0 or negative, break to avoid infinite loop
                if max_Q <= 0
                    fprintf('Warning: No suitable cluster heads found. Using %d cluster heads.\n', tot-1);
                    CH_tot = tot - 1;
                    break;
                end
                
                for i=1:n
                    if(CM(i).Q == max_Q && CM(i).Q>=0)
                        % Relaxed constraints for reclustering
                        if tot == 1  % First CH can be anywhere with energy > 0
                            NET(i).role=1;
                            CH(tot) = NET(i);
                            CM(i).Q = 0;
                            tot =tot+1;
                        elseif tot>1  % Subsequent CHs just need to be far enough from other CHs
                            cl=0;
                            for t = 1:tot-1
                                dts = sqrt((CM(i).x-CH(t).x)^2 + (CM(i).y-CH(t).y)^2);
                                if(dts < 15) % Minimum 15m separation between CHs
                                    cl= cl+1;
                                    break;
                                end
                            end
                            if cl==0
                                NET(i).role=1;
                                CH(tot) = NET(i);
                                CM(i).Q = 0;
                                tot =tot+1;  
                            else
                                CM(i).Q = 0;
                            end
                        end
                        break; % Exit the for loop once we process one node
                    end
                end
                
                if tot > CH_tot
                    break;
                end
            end
            
            % Final check - ensure we have at least one cluster head
            if tot == 1 && aln > 0
                % If no CHs were selected, pick the node with highest energy
                [~, best_idx] = max([NET.E]);
                if NET(best_idx).E > 0
                    NET(best_idx).role = 1;
                    CH(1) = NET(best_idx);
                    CH_tot = 1;
                    fprintf('Emergency: Selected node %d as single cluster head.\n', best_idx);
                end
            end
            %END CLUSTER HEAD ELECTION

            %CLUSTER FORMATION
            for i=1:n
                for ch=1:CH_tot
                    dts_ch(i,ch) = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
                    dts_ch(dts_ch==0)=NaN;
                end
                if NET(i).dts<=range_C
                    NET(i).dest = 0;
                    NET(i).dts_ch = 0;
                else
                    if NET(i).role==0
                        for ch = 1:CH_tot
                            dtsCh = sqrt((NET(i).x-CH(ch).x)^2 + (NET(i).y-CH(ch).y)^2);
                            if NET(i).E>0 && dtsCh == min([dts_ch(i,:)])
                                NET(i).dest = CH(ch).id;
                                NET(i).dts_ch = min([dts_ch(i,:)]);
%                                 figure(1);
%                                 plot([NET(i).x CH(ch).x], [NET(i).y CH(ch).y])
%                                 hold on;
                            end
                        end
                    end  
                end             
            end
            %END CLUSTER FORMATION
            for i=1:CH_tot
                ncm=1;
                for j=1:n
                    if NET(j).E>0
                        if(NET(j).dest == CH(i).id & NET(j).role==0)
                            %cluster(i,ncm)= [];
                            cluster(i,ncm)= NET(j);
                            ncm= ncm+1;
                        end
                    end
                    
                end
            end
            %------------------- END OF RECLUSTERING ---------------------

        end
        CH(j).prev=0;
    end
    %END INTERCLUSTER COMMUNICATION
    
    %Nodes around the sink node
    for l=1:n
        if NET(l).E>0
            if NET(l).dts<=range_C && NET(l).role==0
                if NET(l).prev==0
                    ETx= send_energy;
                    NET(l).E=NET(l).E-ETx-active_energy;
                    energy=energy+ETx+active_energy;
                else
                    Edis = (send_energy + receive_energy + active_energy);
                    NET(l).E = NET(l).E-Edis;
                    energy=energy+Edis;
                    NET(l).prev =0;
                end
                
                %Compute the reward
                Q_old = NET(l).Q;
                R= (p*(NET(l).E - min_E)/(max_E-min_E)+(q1/NET(l).hop));

                %update Q_value
                NET(l).Q =Q_old + alpha*(R+ gamma * maxQ -Q_old) ;
                NET(l).Q = NET(l).Q;
            end
        end
        
        
    end
    
    %Compute round, Energy consumed per round and ...
    rnd = rnd+1;
    E_round(rnd) = energy;
    
    % Track consecutive rounds with zero energy consumption
    if energy == 0
        zero_energy_rounds = zero_energy_rounds + 1;
    else
        zero_energy_rounds = 0;
    end
    
    % Display progress every 50 rounds or when nodes die
    dead=0;
    for i =1:n
        % Mark node as dead if its energy is <= 0 or condition is 0
        % Also mark as dead if energy consumption has been 0 for this round and it's not the first round
        if NET(i).E<=0 || NET(i).cond==0 || (rnd > 1 && energy==0 && op_nodes==1)
            dead = dead+1;
            NET(i).Q= -100;
            NET(i).cond = 0;
            NET(i).E = 0; % Ensure energy is explicitly set to 0
            op_nodes = n-dead;
        end
    end
    dead_rnd(rnd)=dead;
    op_nodes_rnd(rnd)=op_nodes;
    
    % Progress display - show every 50 rounds or when nodes die
    if mod(rnd, 50) == 0 || dead > dead_rnd(max(1, rnd-1)) || rnd <= 10
        fprintf('Round %d: %d nodes alive, %d dead, CH_tot=%d, Energy=%.2f\n', rnd, op_nodes, dead, CH_tot, energy);
    end
end
% END COMMUNICATION PHASE ------------------------------------------------

% Check if simulation ended due to zero energy consumption
if zero_energy_rounds >= 5
    fprintf('Simulation ended due to zero energy consumption for %d consecutive rounds.\n', zero_energy_rounds);
    % Mark any remaining "alive" nodes as dead for final statistics
    op_nodes_rnd(rnd) = 0;
    dead_rnd(rnd) = n;
end

% Save all simulation data to CSV file
results_table = table((1:rnd)', op_nodes_rnd(1:rnd)', dead_rnd(1:rnd)', E_round(1:rnd)', ...
    'VariableNames', {'Round', 'OperatingNodes', 'DeadNodes', 'EnergyConsumed'});

% Create a summary row with key metrics
first_dead_round = find(dead_rnd > 0, 1);
half_dead_round = find(dead_rnd >= n/2, 1);
all_dead_round = find(dead_rnd == n, 1);
if isempty(all_dead_round)
    all_dead_round = rnd; % If not all nodes died, use the last round
end

summary_table = table(n, first_dead_round, half_dead_round, all_dead_round, ...
    'VariableNames', {'TotalNodes', 'FirstNodeDeathRound', 'HalfNodesDeathRound', 'AllNodesDeathRound'});

% Save to CSV files
writetable(results_table, 'simulation_results.csv');
writetable(summary_table, 'simulation_summary.csv');

fprintf('\n-------- Simulation Summary --------\n');
fprintf('Total Nodes: %d\n', n);
fprintf('First node death: Round %d\n', first_dead_round);
fprintf('Half nodes death: Round %d\n', half_dead_round);
fprintf('All nodes death: Round %d\n', all_dead_round);
fprintf('Simulation results saved to CSV files.\n');

% Plotting Simulation Results "Operating Nodes per Transmission" %
figure(2)
plot(1:rnd,op_nodes_rnd(1:rnd),'-','Linewidth',1);
%legend('RL-CEBRP');
title ({'Operating Nodes per Round';'' })
xlabel 'Rounds';
ylabel 'Operating Nodes';
hold on;

% Plotting Simulation Results "Energy consumed per Round" %
figure(3)
plot(1:rnd,E_round(1:rnd),'-','Linewidth',1);
%legend('RL-CEBRP')
title ({'Energy consumed per Round';'' })
xlabel 'Rounds';
ylabel 'Energy consumed in J';
hold on;

% % Plotting Simulation Results "Energy consumed per Round" %
figure(4)
plot(1:rnd,E_round(1:rnd),'-r','Linewidth',2);
%legend('RL-EBRP')
title ({'EER-RL'; 'Energy consumed per Round';})
xlabel 'Rounds';
ylabel 'Energy consumed in J';
hold on;
% 
% % Plotting Simulation Results "Cumulated dead nodes per Round" %
figure(5)
plot(1:rnd,dead_rnd(1:rnd),'-r','Linewidth',2);
%legend('RL-EBRP');
title ({'EER-RL'; 'Total dead nodes per Round';})
xlabel 'Rounds';
ylabel 'Dead Nodes';
hold on;