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

% Regular Node Algorithm
\begin{algorithm}[H]
\caption{Regular Node Operation}
\label{alg:regular_node}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, current time $t$, sensor data $D_t$
\ENSURE Data transmission decision and mode update

\STATE \textbf{Step 1: Sleep Scheduling}
\STATE $send\_permission \leftarrow$ Algorithm \ref{alg:data_transmission}($D_t$)
\STATE $mode \leftarrow$ Algorithm \ref{alg:sleep_scheduling}($send\_permission$, $t$)

\IF{$mode = SLEEP$}
    \STATE Consume sleep energy: $E_i \leftarrow E_i - P_{sleep}$
    \RETURN \COMMENT{Skip transmission}
\ENDIF

\STATE \textbf{Step 2: Data Processing (ACTIVE mode)}
\STATE Read sensor data $D_t$
\STATE Consume active energy: $E_i \leftarrow E_i - P_{active}$

\IF{$send\_permission = TRUE$}
    \STATE \textbf{Step 3: Routing Decision}
    \STATE $next\_hop \leftarrow$ DQN routing decision (Algorithm \ref{alg:dqn_simple})
    
    \IF{$next\_hop$ is available}
        \STATE Create packet $P$ with data $D_t$
        \STATE Send $P$ to $next\_hop$
        \STATE Consume transmission energy: $E_i \leftarrow E_i - P_{send}$
        \STATE Update DQN with reward
    \ENDIF
\ENDIF

\end{algorithmic}
\end{algorithm}

% Cluster Head Algorithm
\begin{algorithm}[H]
\caption{Cluster Head Operation}
\label{alg:cluster_head}
\begin{algorithmic}[1]

\REQUIRE Cluster Head $CH_j$, packet queue $Q_j$
\ENSURE Data aggregation and forwarding

\STATE \textbf{Step 1: Process Incoming Packets}
\WHILE{$Q_j$ is not empty}
    \STATE $packet \leftarrow Q_j.dequeue()$
    
    \IF{packet is from cluster member}
        \STATE $aggregated\_data \leftarrow$ Aggregate($packet.data$)
        \STATE Store aggregated data
    \ELSIF{packet is from other CH}
        \STATE Update neighbor table with packet info
    \ENDIF
    
    \STATE \textbf{Step 2: Forward to Sink}
    \STATE Create new packet with aggregated data
    \STATE Consume transmission energy: $E_j \leftarrow E_j - P_{send}$
    
    \IF{Sink is within range}
        \STATE Send directly to Sink
    \ELSE
        \STATE Route through nearest CH toward Sink
    \ENDIF
\ENDWHILE

\STATE \textbf{Step 3: Additional CH Energy Cost}
\STATE $E_j \leftarrow E_j - P_{active} \times 0.02$ \COMMENT{CH overhead}

\end{algorithmic}
\end{algorithm}

% Sleep Scheduling Algorithm
\begin{algorithm}[H]
\caption{Adaptive Sleep Scheduling}
\label{alg:sleep_scheduling}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, send permission status, current time $t$
\ENSURE Updated node mode (ACTIVE/SLEEP)

\STATE \textbf{Check Wake-up Condition:}
\IF{$mode_i = SLEEP$ AND $t \geq sleep\_until_i$}
    \STATE $mode_i \leftarrow ACTIVE$
    \STATE $no\_send\_count_i \leftarrow 0$
\ENDIF

\STATE \textbf{Update Send Counter:}
\IF{send permission = FALSE}
    \STATE $no\_send\_count_i \leftarrow no\_send\_count_i + 1$
\ELSE
    \STATE $no\_send\_count_i \leftarrow 0$
\ENDIF

\STATE \textbf{Sleep Decision:}
\STATE $threshold_i \leftarrow base\_threshold + (node\_id \bmod 5)$

\IF{$mode_i = ACTIVE$ AND $no\_send\_count_i \geq threshold_i$}
    \STATE $mode_i \leftarrow SLEEP$
    \STATE $sleep\_duration \leftarrow calculate\_sleep\_duration(node\_id)$
    \STATE $sleep\_until_i \leftarrow t + sleep\_duration$
\ENDIF

\RETURN $mode_i$

\end{algorithmic}
\end{algorithm}

% Data Transmission Decision Algorithm
\begin{algorithm}[H]
\caption{Restrict Data Transmission}
\label{alg:data_transmission}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, sensor data $D_t$, change threshold $\theta$
\ENSURE Send permission (TRUE/FALSE)

\IF{$last\_data_i = NULL$} \COMMENT{First reading}
    \STATE $last\_data_i \leftarrow D_t$
    \RETURN TRUE
\ENDIF

\STATE \textbf{Node-specific sensitivity:}
\STATE $sensitivity_i \leftarrow 1 + (node\_id \bmod 6) \times 0.3$
\STATE $adjusted\_threshold \leftarrow \theta \times sensitivity_i$

\STATE \textbf{Change detection:}
\STATE $data\_change \leftarrow |D_t - last\_data_i|$

\IF{$data\_change > adjusted\_threshold$}
    \STATE $last\_data_i \leftarrow D_t$
    \RETURN TRUE
\ELSE
    \STATE \textbf{Probabilistic transmission:} \COMMENT{Break synchronization}
    \IF{$random() < 0.02$}
        \RETURN TRUE
    \ENDIF
    \RETURN FALSE
\ENDIF

\end{algorithmic}
\end{algorithm>

% Simplified DQN Routing
\begin{algorithm}[H]
\caption{DQN-Based Routing (Simplified)}
\label{alg:dqn_simple}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, available neighbors
\ENSURE Next hop selection

\STATE \textbf{State extraction:}
\STATE $s_t \leftarrow [energy\_ratio, ch\_distance, sink\_distance, urgency]$

\STATE \textbf{Action selection ($\epsilon$-greedy):}
\IF{$random() > \epsilon_t$}
    \STATE $action \leftarrow \arg\max_a Q(s_t, a)$ \COMMENT{Exploitation}
\ELSE
    \STATE $action \leftarrow random\_action()$ \COMMENT{Exploration}
\ENDIF

\STATE \textbf{Map action to next hop:}
\IF{$action = 0$} \COMMENT{Forward to CH}
    \STATE $next\_hop \leftarrow cluster\_head_i$
\ELSIF{$action = 1$} \COMMENT{Forward to Sink}
    \STATE $next\_hop \leftarrow sink$ (if in range)
\ELSE \COMMENT{Sleep}
    \STATE $next\_hop \leftarrow NULL$
\ENDIF

\STATE \textbf{Learning update:}
\STATE $reward \leftarrow calculate\_reward(energy, distance, success)$
\STATE Store experience $(s_{t-1}, a_{t-1}, reward, s_t)$ in memory
\STATE Update Q-network with experience replay

\RETURN $next\_hop$

\end{algorithmic}
\end{algorithm>

% Reward Calculation
\begin{algorithm}[H]
\caption{Routing Reward Calculation}
\label{alg:reward_calc}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, neighbor $n_j$, transmission success
\ENSURE Calculated reward value

\STATE \textbf{Base reward calculation:}
\IF{transmission failed}
    \RETURN $-1.0$
\ENDIF

\STATE $base\_reward \leftarrow 0.5$

\STATE \textbf{Energy efficiency component:}
\STATE $energy\_reward \leftarrow 0.5 \times \frac{E_i}{E_{initial}}$

\STATE \textbf{Network lifetime component:}
\STATE $alive\_ratio \leftarrow \frac{|\{n : E_n > 0\}|}{N}$
\STATE $lifetime\_reward \leftarrow 0.5 \times alive\_ratio$

\STATE \textbf{Distance-based component:}
\STATE $d \leftarrow distance(n_i, n_j)$
\STATE $normalized\_distance \leftarrow \frac{d}{send\_range}$
\STATE $distance\_reward \leftarrow 0.5 \times (1 - normalized\_distance)$

\STATE $total\_reward \leftarrow base\_reward + energy\_reward + lifetime\_reward + distance\_reward$

\RETURN $total\_reward$

\end{algorithmic}
\end{algorithm>

% Q-Value Update (Traditional RL Component)
\begin{algorithm}[H]
\caption{Q-Value Update (RLBR Component)}
\label{alg:q_update}
\begin{algorithmic}[1]

\REQUIRE Current node $n_i$, neighbor $n_j$, learning rate $\alpha$
\ENSURE Updated Q-value

\STATE \textbf{Calculate reward:}
\STATE $n \leftarrow calculate\_n\_parameter(d_{normalized})$
\STATE $reward \leftarrow \frac{E_j}{d_{ij}^n \times h_j}$

\WHERE:
\STATE $E_j$ = energy of neighbor $j$
\STATE $d_{ij}$ = distance between nodes $i$ and $j$  
\STATE $h_j$ = hop count from neighbor $j$ to sink
\STATE $n$ = distance factor parameter

\STATE \textbf{Q-value update:}
\STATE $Q_{new}(i,j) \leftarrow (1-\alpha) \times Q_{old}(i,j) + \alpha \times (reward + Q(j))$

\WHERE:
\STATE $Q(j) \leftarrow \max_{k} Q(j,k)$ for all neighbors $k$ of $j$

\RETURN $Q_{new}(i,j)$

\end{algorithmic}
\end{algorithm>

% Energy Management (Simplified)
\begin{algorithm}[H]
\caption{Energy Management (Simplified)}
\label{alg:energy_simple}
\begin{algorithmic}[1]

\REQUIRE Node $n_i$, activity type
\ENSURE Updated energy level

\STATE \textbf{Base energy consumption:}
\IF{activity = "send"}
    \STATE $cost \leftarrow 0.3$ J
\ELSIF{activity = "receive"}
    \STATE $cost \leftarrow 0.2$ J
\ELSIF{activity = "active"}
    \STATE $cost \leftarrow 0.1$ J/s
\ELSIF{activity = "sleep"}
    \STATE $cost \leftarrow 0.05$ J/s
\ENDIF

\STATE \textbf{Node-specific variation:}
\STATE $efficiency \leftarrow 1 + (node\_id \bmod 7) \times 0.05$
\STATE $cost \leftarrow cost \times efficiency$

\STATE \textbf{Additional CH overhead:}
\IF{node is Cluster Head}
    \STATE $cost \leftarrow cost + 0.002$ J \COMMENT{2\% overhead}
\ENDIF

\STATE $E_i \leftarrow \max(0, E_i - cost)$
\RETURN $E_i > 0$

\end{algorithmic}
\end{algorithm>

% System Parameters Table
\begin{table}[h]
\centering
\caption{PSWR-DRL System Parameters}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Category} & \textbf{Parameter} & \textbf{Value} \\
\hline
Network & Nodes & 10, 30, 50, 100 \\
& Clusters & 4, 8, 12, 20 \\
& Send Range & 10-20 meters \\
& Area & 60×60 to 100×100 m \\
\hline
Energy & Initial Energy & 100-150 Joules \\
& Send Power & 0.3 J/transmission \\
& Receive Power & 0.2 J/reception \\
& Active Power & 0.1 J/second \\
& Sleep Power & 0.05 J/second \\
\hline
DQN & Learning Rate & 0.001 \\
& Discount Factor & 0.99 \\
& Exploration & 0.9 → 0.05 \\
& Batch Size & 32 \\
& Memory Size & 10,000 \\
\hline
Sleep & Base Threshold & 5-10 \\
& Duration & 30 seconds \\
& Node Variation & ID-based \\
\hline
Data & Change Threshold & 1.5-10.0 \\
& Period & 6 seconds \\
& Sensitivity & Node-specific \\
\hline
\end{tabular}
\end{table>

\end{document}
```

## Compact Version for Thesis

```latex
% Ultra-Compact Version for Thesis Overview
\begin{algorithm}[H]
\caption{PSWR-DRL: Complete System (Compact)}
\label{alg:pswr_drl_compact}
\begin{algorithmic}[1]

\REQUIRE WSN with $N$ nodes, $K$ clusters
\ENSURE Optimized energy and network lifetime

\STATE \textbf{Initialize:} Deploy nodes, select CHs, setup DQN
\FOR{$t = 1$ to $T_{max}$}
    \FOR{each node $n_i$}
        \STATE $data \leftarrow$ read sensor
        \STATE $send\_ok \leftarrow$ data change $>$ threshold
        \STATE $mode \leftarrow$ sleep scheduling($send\_ok$)
        
        \IF{$mode = ACTIVE$ AND $send\_ok$}
            \STATE $next\_hop \leftarrow$ DQN action selection
            \STATE Send packet to $next\_hop$
            \STATE Update DQN with reward
        \ENDIF
        \STATE Update energy based on activity
    \ENDFOR
    \IF{$t \bmod 300 = 0$} Rotate cluster heads \ENDIF
\ENDFOR

\end{algorithmic}
\end{algorithm>
```

## Key Simplifications Made:

### 1. **Reduced Algorithm Complexity:**
- Combined multiple complex algorithms into simple, focused ones
- Removed redundant state calculations
- Simplified DQN to 4-dimensional state space
- Streamlined energy management

### 2. **Clear Algorithm Flow:**
- Main algorithm calls sub-algorithms in logical order
- Each algorithm has single, clear purpose
- Removed nested complexity and excessive parameters

### 3. **Focused on Core Concepts:**
- **Sleep Scheduling**: Simple counter-based approach
- **Data Transmission**: Change threshold with node diversity
- **DQN Routing**: Basic ε-greedy with simple state
- **Energy Management**: Straightforward consumption model

### 4. **Matches Your Implementation:**
- Algorithms directly reflect your actual code structure
- Uses same parameter names and values
- Follows the reference approach you provided
- Maintains the RLBEEP/RLBR hybrid nature

### 5. **Professional Presentation:**
- Clean LaTeX formatting
- Proper mathematical notation
- Compact parameter table
- Ready for thesis inclusion

This simplified version captures the essence of your PSWR-DRL system while being much more readable and understandable, perfect for academic presentation in your thesis!
