# PSWR-DRL: Simplified System Algorithms (LaTeX Format)

## Complete Algorithm Suite in LaTeX

```latex
\documentclass{article}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

\title{PSWR-DRL: Power Saving Wireless Routing with Deep Reinforcement Learning}
\author{Simplified Algorithm Suite}
\maketitle

% Main RLBEEP Algorithm
\begin{algorithm}[H]
\caption{RLBEEP: Main System Algorithm}
\label{alg:rlbeep_main}
\begin{algorithmic}[1]

\REQUIRE Network nodes $N$, Clusters $K$, Initial energy $E_0$
\ENSURE Maximized network lifetime and energy efficiency

\STATE \textbf{Initialize Network:}
\STATE Deploy $N$ nodes in strategic grid positions
\STATE Select $K$ cluster heads based on energy and coverage
\STATE Assign nodes to nearest cluster heads
\STATE Initialize DQN agents for each node

\WHILE{network is operational AND $t < T_{max}$}
    \FOR{each alive node $n_i$}
        \IF{$n_i$ is Regular Node}
            \STATE Execute Algorithm \ref{alg:regular_node}
        \ELSIF{$n_i$ is Cluster Head}
            \STATE Execute Algorithm \ref{alg:cluster_head}
        \ENDIF
        \STATE Update energy consumption
    \ENDFOR
    
    \IF{$t \bmod 300 = 0$} 
        \STATE Rotate cluster heads based on energy levels
    \ENDIF
    
    \STATE $t \leftarrow t + 1$
\ENDWHILE

\RETURN Network statistics and performance metrics

\end{algorithmic}
\end{algorithm}

% Cluster Formation Algorithm
\begin{algorithm}[H]
\caption{Strategic Cluster Formation}
\label{alg:cluster_formation}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Node set $V$, number of clusters $K$, area dimensions $(L, W)$
\STATE \textbf{Output:} Cluster assignments and cluster heads

\STATE Generate strategic node positions using modified grid placement
\STATE $grid\_spacing_x \leftarrow L / (\sqrt{N} + 1)$
\STATE $grid\_spacing_y \leftarrow W / (\sqrt{N} + 1)$

\FOR{$i = 1$ to $N$}
    \STATE $row \leftarrow \lfloor i / \sqrt{N} \rfloor$
    \STATE $col \leftarrow i \bmod \sqrt{N}$
    \STATE $x_i \leftarrow col \cdot grid\_spacing_x + random(-5, 5)$
    \STATE $y_i \leftarrow row \cdot grid\_spacing_y + random(-5, 5)$
    \STATE Place node $v_i$ at position $(x_i, y_i)$
\ENDFOR

\STATE \textbf{Select Optimal Cluster Heads:}
\STATE $region\_width \leftarrow L / \sqrt{K}$
\STATE $region\_height \leftarrow W / \sqrt{K}$

\FOR{$j = 1$ to $K$}
    \STATE $region\_x \leftarrow (j-1) \bmod \sqrt{K} \cdot region\_width$
    \STATE $region\_y \leftarrow \lfloor (j-1) / \sqrt{K} \rfloor \cdot region\_height$
    \STATE Find node closest to region center as $CH_j$
    \STATE Convert selected node to cluster head
\ENDFOR

\STATE \textbf{Assign Nodes to Clusters:}
\FOR{each node $v_i \in V$}
    \STATE $closest\_ch \leftarrow \arg\min_{CH_j} distance(v_i, CH_j)$
    \IF{$distance(v_i, closest\_ch) \leq send\_range$}
        \STATE Assign $v_i$ to cluster of $closest\_ch$
    \ENDIF
\ENDFOR

\end{algorithmic}
\end{algorithm}

% DQN Initialization Algorithm
\begin{algorithm}[H]
\caption{Deep Q-Network Initialization}
\label{alg:dqn_init}
\begin{algorithmic}[1]

\STATE \textbf{Input:} State dimension $d_s = 9$, action dimension $d_a = 3$
\STATE \textbf{Output:} Initialized DQN agent for each node

\STATE \textbf{State Space Definition:} $S = [s_1, s_2, ..., s_9]$ where:
\STATE $s_1$: Normalized energy level $= E_{current} / E_{initial}$
\STATE $s_2$: Normalized distance to cluster head
\STATE $s_3$: Normalized distance to sink
\STATE $s_4$: Normalized hop count to sink
\STATE $s_5$: Data transmission urgency
\STATE $s_6$: Network congestion (queue length)
\STATE $s_7$: Sleep pressure (no-send count)
\STATE $s_8$: Cluster health (CH energy level)
\STATE $s_9$: Time factor (diurnal pattern)

\STATE \textbf{Action Space Definition:} $A = \{a_0, a_1, a_2\}$ where:
\STATE $a_0$: Forward packet to cluster head
\STATE $a_1$: Forward packet directly to sink
\STATE $a_2$: Enter sleep mode (no transmission)

\STATE \textbf{Neural Network Architecture:}
\STATE Policy Network: $\pi_\theta: S \rightarrow A$ with layers $[d_s, 64, 64, d_a]$
\STATE Target Network: $\pi_{\theta'}: S \rightarrow A$ (copy of policy network)
\STATE Activation: ReLU for hidden layers
\STATE Optimizer: Adam with learning rate $\alpha = 0.001$

\STATE \textbf{Experience Replay:}
\STATE Initialize replay memory $\mathcal{D}$ with capacity $C = 10000$
\STATE Initialize exploration parameters: $\epsilon_{start} = 0.9$, $\epsilon_{end} = 0.05$

\end{algorithmic}
\end{algorithm}

% Adaptive Sleep Scheduling Algorithm
\begin{algorithm}[H]
\caption{Node-Specific Adaptive Sleep Scheduling}
\label{alg:sleep_scheduling}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Node $v_i$, current time $t$, send permission $p_{send}$
\STATE \textbf{Output:} Updated node mode (ACTIVE/SLEEP)

\STATE \textbf{Check Wake-up Condition:}
\IF{$mode(v_i) = SLEEP$ AND $t \geq sleep\_until(v_i)$}
    \STATE $mode(v_i) \leftarrow ACTIVE$
    \STATE $no\_send\_count(v_i) \leftarrow 0$
\ENDIF

\STATE \textbf{Update No-Send Counter:}
\IF{$p_{send} = false$}
    \STATE $no\_send\_count(v_i) \leftarrow no\_send\_count(v_i) + 1$
\ELSE
    \STATE $no\_send\_count(v_i) \leftarrow 0$
\ENDIF

\STATE \textbf{Node-Specific Sleep Decision:}
\STATE $node\_sleep\_threshold \leftarrow \tau_{sleep} + (id(v_i) \bmod 5)$
\STATE $sleep\_variation \leftarrow 1 + (id(v_i) \bmod 7) \times 0.15$

\IF{$mode(v_i) = ACTIVE$ AND $no\_send\_count(v_i) \geq node\_sleep\_threshold$}
    \STATE $mode(v_i) \leftarrow SLEEP$
    \STATE $sleep\_duration \leftarrow \tau_{sleep} \times sleep\_variation$
    \STATE $sleep\_duration \leftarrow sleep\_duration + random(-5, 5)$
    \STATE $sleep\_duration \leftarrow \max(10, sleep\_duration)$
    \STATE $sleep\_until(v_i) \leftarrow t + sleep\_duration$
\ENDIF

\STATE \textbf{Energy Consumption Based on Mode:}
\IF{$mode(v_i) = SLEEP$}
    \STATE $Energy(v_i) \leftarrow Energy(v_i) - P_{sleep} \times \eta_{node}$
\ELSE
    \STATE $Energy(v_i) \leftarrow Energy(v_i) - P_{active} \times \eta_{node}$
\ENDIF

\end{algorithmic}
\end{algorithm}

% Data-Driven Transmission Algorithm
\begin{algorithm}[H]
\caption{Intelligent Data-Driven Transmission}
\label{alg:data_transmission}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Node $v_i$, sensor data $D_t$, change threshold $\theta_{change}$
\STATE \textbf{Output:} Transmission decision $p_{transmit}$

\STATE \textbf{Get Current Sensor Reading:}
\STATE $current\_data \leftarrow getSensorReading(v_i, t \bmod \tau_{period})$
\STATE Add $current\_data$ to $data\_history(v_i)$

\STATE \textbf{Node-Specific Sensitivity Adjustment:}
\STATE $node\_sensitivity \leftarrow 1 + (id(v_i) \bmod 6) \times 0.3$
\STATE $adjusted\_threshold \leftarrow \theta_{change} \times node\_sensitivity$

\STATE \textbf{Change Detection:}
\IF{$last\_data(v_i) = null$}
    \STATE $last\_data(v_i) \leftarrow current\_data$
    \RETURN $true$ \COMMENT{First transmission}
\ENDIF

\STATE $data\_difference \leftarrow |current\_data - last\_data(v_i)|$

\STATE \textbf{Transmission Decision:}
\IF{$data\_difference > adjusted\_threshold$}
    \STATE $p_{transmit} \leftarrow true$
    \STATE $last\_data(v_i) \leftarrow current\_data$
\ELSE
    \STATE $p_{transmit} \leftarrow false$
    \STATE \textbf{Probabilistic Transmission:} \COMMENT{Break synchronization}
    \IF{$random() < 0.02$}
        \STATE $p_{transmit} \leftarrow true$
    \ENDIF
\ENDIF

\RETURN $p_{transmit}$

\end{algorithmic}
\end{algorithm}

% DQN-Based Routing Algorithm
\begin{algorithm}[H]
\caption{DQN-Based Intelligent Routing}
\label{alg:dqn_routing}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Node $v_i$, packet $P$, destination $d$
\STATE \textbf{Output:} Next hop selection and Q-value update

\STATE \textbf{State Representation:}
\STATE $s_t \leftarrow getEnhancedState(v_i)$ \COMMENT{9-dimensional state vector}
\STATE $s_t[0] \leftarrow Energy(v_i) / E_{initial}(v_i)$
\STATE $s_t[1] \leftarrow \min(1.0, distance(v_i, CH) / send\_range)$
\STATE $s_t[2] \leftarrow \min(1.0, distance(v_i, sink) / (L + W))$
\STATE $s_t[3] \leftarrow \min(1.0, hopCount(v_i, sink) / 10.0)$
\STATE $s_t[4] \leftarrow \min(1.0, |current\_data - last\_data| / \theta_{change})$
\STATE $s_t[5] \leftarrow \min(1.0, |send\_queue(v_i)| / 10.0)$
\STATE $s_t[6] \leftarrow \min(1.0, no\_send\_count(v_i) / \tau_{sleep})$
\STATE $s_t[7] \leftarrow Energy(CH) / E_{initial}(CH)$ if CH exists
\STATE $s_t[8] \leftarrow (t \bmod 300) / 300.0$ \COMMENT{Diurnal pattern}

\STATE \textbf{Action Selection (ε-greedy):}
\STATE $\epsilon_t \leftarrow \epsilon_{min} + (\epsilon_{max} - \epsilon_{min}) \times e^{-steps / \tau_{decay}}$

\IF{$random() > \epsilon_t$}
    \STATE $a_t \leftarrow \arg\max_a Q_\theta(s_t, a)$ \COMMENT{Exploitation}
\ELSE
    \STATE $a_t \leftarrow randomAction()$ \COMMENT{Exploration}
\ENDIF

\STATE \textbf{Action Execution:}
\IF{$a_t = 0$} \COMMENT{Forward to Cluster Head}
    \IF{$cluster\_id(v_i) \neq null$}
        \STATE $next\_hop \leftarrow CH_{cluster\_id(v_i)}$
    \ENDIF
\ELSIF{$a_t = 1$} \COMMENT{Forward to Sink}
    \IF{$sink \in neighbors(v_i)$}
        \STATE $next\_hop \leftarrow sink$
    \ENDIF
\ELSE \COMMENT{Sleep/No transmission}
    \STATE $next\_hop \leftarrow null$
\ENDIF

\STATE \textbf{Reward Calculation:}
\STATE $r_t \leftarrow calculateReward(v_i, a_t, transmission\_success)$
\STATE $r_t \leftarrow 0.5 + 0.5 \times (Energy(v_i) / E_{initial})$ \COMMENT{Energy efficiency}
\STATE $r_t \leftarrow r_t + 0.5 \times (alive\_nodes / total\_nodes)$ \COMMENT{Network lifetime}
\STATE $r_t \leftarrow r_t + distance\_based\_reward(v_i, next\_hop)$

\STATE \textbf{Experience Storage and Learning:}
\STATE Store $(s_{t-1}, a_{t-1}, r_t, s_t, done)$ in replay memory $\mathcal{D}$
\IF{$|\mathcal{D}| \geq batch\_size$}
    \STATE Sample minibatch from $\mathcal{D}$
    \STATE Compute target: $y = r + \gamma \max_{a'} Q_{\theta'}(s', a')$
    \STATE Update policy network: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(y, Q_\theta(s,a))$
\ENDIF

\IF{$t \bmod \tau_{target\_update} = 0$}
    \STATE $\theta' \leftarrow \theta$ \COMMENT{Update target network}
\ENDIF

\end{algorithmic}
\end{algorithm}

% Cluster Head Rotation Algorithm
\begin{algorithm}[H]
\caption{Energy-Aware Cluster Head Rotation}
\label{alg:ch_rotation}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Current cluster heads $CH_{set}$, rotation interval $\tau_{rotation}$
\STATE \textbf{Output:} Updated cluster head assignments

\IF{$t \bmod \tau_{rotation} \neq 0$ OR $t < \tau_{rotation}$}
    \RETURN \COMMENT{Skip rotation}
\ENDIF

\STATE \textbf{Preserve Connectivity:}
\STATE $connected\_CHs \leftarrow \{CH_i : distance(CH_i, sink) \leq send\_range\}$
\STATE $preserve\_CH \leftarrow \arg\max_{CH \in connected\_CHs} Energy(CH)$

\STATE \textbf{Select New Cluster Heads:}
\FOR{each cluster $j$}
    \IF{$CH_j = preserve\_CH$}
        \STATE Keep current cluster head \COMMENT{Maintain connectivity}
    \ELSE
        \STATE Convert $CH_j$ back to regular node
        \STATE $candidate\_members \leftarrow \{v_i : cluster\_id(v_i) = j \land Energy(v_i) > 0\}$
        \STATE $new\_CH \leftarrow \arg\max_{v \in candidate\_members} Energy(v)$
        \STATE Convert $new\_CH$ to cluster head
        \STATE Update cluster head list
    \ENDIF
\ENDFOR

\STATE \textbf{Re-establish Network Topology:}
\STATE Reassign nodes to clusters based on proximity
\STATE Recalculate hop counts to sink
\STATE Verify network connectivity

\STATE \textbf{Connectivity Recovery:}
\IF{no cluster head connected to sink}
    \STATE Find closest alive node to sink
    \STATE Convert to cluster head if within reasonable range
    \STATE Update network topology accordingly
\ENDIF

\end{algorithmic}
\end{algorithm}

% Energy Management Algorithm
\begin{algorithm}[H]
\caption{Heterogeneous Energy Management}
\label{alg:energy_management}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Node $v_i$, activity type $activity$, time ratio $t_{ratio}$
\STATE \textbf{Output:} Updated energy level

\STATE \textbf{Base Energy Consumption:}
\IF{$activity = "send"$}
    \STATE $E_{cost} \leftarrow P_{send}$
\ELSIF{$activity = "receive"$}
    \STATE $E_{cost} \leftarrow P_{receive}$
\ELSIF{$activity = "active"$}
    \STATE $E_{cost} \leftarrow P_{active}$
\ELSIF{$activity = "sleep"$}
    \STATE $E_{cost} \leftarrow P_{sleep}$
\ENDIF

\STATE \textbf{Node-Specific Energy Efficiency:}
\STATE $node\_efficiency \leftarrow 1 + (id(v_i) \bmod 7) \times 0.05$
\STATE $E_{cost} \leftarrow E_{cost} \times node\_efficiency$

\STATE \textbf{Base Energy Loss (Hardware Leakage):}
\STATE $base\_loss \leftarrow P_{active} \times 0.001$
\STATE $node\_base\_factor \leftarrow 1 + (id(v_i) \bmod 5) \times 0.2$
\STATE $base\_loss \leftarrow base\_loss \times node\_base\_factor$

\STATE \textbf{Time-Based Degradation:}
\STATE $degradation\_factor \leftarrow 1 + (t_{ratio} \times 0.01)$
\STATE $base\_loss \leftarrow base\_loss \times degradation\_factor$

\STATE \textbf{Apply Energy Loss Based on Activity:}
\IF{$activity = "sleep"$}
    \STATE $total\_cost \leftarrow E_{cost} + base\_loss \times 0.05$
\ELSE
    \STATE $total\_cost \leftarrow E_{cost} + base\_loss$
\ENDIF

\STATE \textbf{Additional Cost for Cluster Heads:}
\IF{$type(v_i) = CLUSTER\_HEAD$}
    \STATE $extra\_cost \leftarrow P_{active} \times 0.02$
    \STATE $extra\_cost \leftarrow extra\_cost \times (1 + t_{ratio} \times 0.05)$
    \STATE $total\_cost \leftarrow total\_cost + extra\_cost$
\ENDIF

\STATE \textbf{Update Energy:}
\STATE $Energy(v_i) \leftarrow \max(0, Energy(v_i) - total\_cost)$
\RETURN $Energy(v_i) > 0$

\end{algorithmic}
\end{algorithm}

% Performance Metrics Calculation
\begin{algorithm}[H]
\caption{Performance Metrics Calculation}
\label{alg:performance_metrics}
\begin{algorithmic}[1]

\STATE \textbf{Input:} Simulation results over time $T$
\STATE \textbf{Output:} Performance metrics

\STATE \textbf{First Node Death (FND) Time:}
\STATE $FND \leftarrow \min\{t : \exists v_i \in V, Energy(v_i, t) = 0\}$

\STATE \textbf{Network Lifetime:}
\STATE $alive\_nodes(t) \leftarrow |\{v_i \in V : Energy(v_i, t) > 0\}|$
\STATE $network\_lifetime \leftarrow \max\{t : alive\_nodes(t) \geq \lceil 0.1 \times N \rceil\}$

\STATE \textbf{Energy Efficiency:}
\STATE $total\_consumed \leftarrow \sum_{i=1}^N (E_{initial} - Energy(v_i, T))$
\STATE $total\_transmissions \leftarrow \sum_{t=0}^T transmissions(t)$
\STATE $energy\_efficiency \leftarrow total\_transmissions / total\_consumed$

\STATE \textbf{Data Delivery Ratio:}
\STATE $packets\_sent \leftarrow \sum_{i=1}^N transmit\_count(v_i)$
\STATE $packets\_received \leftarrow |received\_packets(sink)|$
\STATE $delivery\_ratio \leftarrow packets\_received / packets\_sent$

\STATE \textbf{Network Coverage:}
\STATE $coverage(t) \leftarrow alive\_nodes(t) / N \times 100\%$

\RETURN $\{FND, network\_lifetime, energy\_efficiency, delivery\_ratio, coverage\}$

\end{algorithmic}
\end{algorithm}

% Main System Parameters
\begin{table}[h]
\centering
\caption{PSWR-DRL System Parameters}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Category} & \textbf{Parameter} & \textbf{Value} \\
\hline
\multirow{4}{*}{Network} & Nodes (N) & 10, 30, 50, 100 \\
& Clusters (K) & 4, 8, 12, 20 \\
& Send Range & 10 meters \\
& Area & 60×60 meters \\
\hline
\multirow{5}{*}{Energy} & Initial Energy & 100 Joules \\
& Send Power & 0.3 J/transmission \\
& Receive Power & 0.2 J/reception \\
& Active Power & 0.1 J/second \\
& Sleep Power & 0.05 J/second \\
\hline
\multirow{6}{*}{DQN} & Learning Rate (α) & 0.001 \\
& Discount Factor (γ) & 0.99 \\
& Exploration (ε) & 0.9 → 0.05 \\
& Batch Size & 32 \\
& Memory Size & 10,000 \\
& Target Update & Every 10 steps \\
\hline
\multirow{4}{*}{Sleep Scheduling} & Sleep Threshold & 5 + (node\_id \% 5) \\
& Sleep Duration & 30 × (1 + variation) \\
& Wake Interval & 30 seconds \\
& Variation Factor & node\_id \% 7 × 0.15 \\
\hline
\multirow{3}{*}{Data Transmission} & Change Threshold & 1.5 × sensitivity \\
& Transmission Period & 6 seconds \\
& Sensitivity Factor & 1 + (node\_id \% 6) × 0.3 \\
\hline
\end{tabular}
\end{table}

\end{document}
```

## Alternative Compact Version

```latex
% Compact version for thesis inclusion
\begin{algorithm}[H]
\caption{PSWR-DRL: Power Saving Wireless Routing with Deep Reinforcement Learning}
\label{alg:pswr_drl_main}
\begin{algorithmic}[1]

\REQUIRE Network $G=(V,E)$, Energy $E_0$, DQN parameters $\{\alpha, \gamma, \epsilon\}$
\ENSURE Maximized network lifetime and energy efficiency

\STATE \textbf{Initialize:} Deploy nodes, form clusters, setup DQN agents
\FOR{$t = 0$ to $T_{max}$}
    \FOR{each alive node $v_i \in V$}
        \STATE $sensor\_data \leftarrow$ Read sensor value
        \STATE $state \leftarrow$ Extract 9D state vector
        \STATE $action \leftarrow$ DQN policy ($\epsilon$-greedy)
        
        \IF{should\_transmit($sensor\_data$, $threshold$)}
            \IF{$action = 0$} \COMMENT{Route to CH}
                \STATE Forward packet to cluster head
            \ELSIF{$action = 1$} \COMMENT{Route to Sink}
                \STATE Forward packet directly to sink
            \ELSE \COMMENT{Sleep mode}
                \STATE Enter adaptive sleep state
            \ENDIF
            \STATE $reward \leftarrow$ Calculate routing reward
            \STATE Update DQN with experience $(s,a,r,s')$
        \ENDIF
        
        \STATE Update energy based on activity
        \STATE Update sleep scheduling state
    \ENDFOR
    
    \IF{$t \bmod rotation\_interval = 0$}
        \STATE Rotate cluster heads based on energy
    \ENDIF
    
    \STATE Record network statistics
\ENDFOR

\RETURN Network lifetime, FND time, energy efficiency

\end{algorithmic}
\end{algorithm}
```
