# Defense Presentation Script (Slide by Slide)

## Slide 1 – Title Slide
Assalamualaikum, respected teachers and audience.  
Today we would like to present our thesis work titled by *PSR-DRL: Power Saving Routing in Wireless Sensor Networks using Deep Reinforcement Learning.*

I am Farhan Ishtiyak Sezar, presenting together with my team-mate Rashedunnabi. We are honored to be supervised by our respected Dr. Aminul Islam sir.

---

## Slide 2 – Presentation Outline
Here is our presentation outline.  
We will discuss introduction, related works, research gap, problem statement, our contributions, system model, simulation results, and conclusion.  

---
## Slide 3 – Introduction
A Wireless Sensor Network is a group of many small, battery-powered nodes.
They sense data like temperature or motion, process it, and send it to a base station.
They work without internet or cables, which makes them useful in disaster situations when other networks fail.
 

---

## Slide 4 – Introduction (continued)
WSNs are used in many areas like environment monitoring, smart cities, industries, and healthcare.
But they face big problems like small battery life, hard maintenance, and static routing that cannot adapt to changes. 

---

## Slide 5 – Related Works (Traditional Power Saving Routing)
This table shows traditional power saving routing methods.  
They use techniques like flat routing, cluster head rotation, and chain-based routing.  
These methods are simple, but they suffer from poor adaptability, uneven energy use, lack of scalability, and weak performance in dynamic network conditions. 

---

## Slide 6 – Related Works (Machine Learning Based)
Here we present machine learning based routing methods.  
They use reinforcement learning and neural networks for better energy distribution.  
These ML methods are better than traditional ones but still have complexity issues, network lifetime not up to the mark, and limited scalability for large networks.  

---

## Slide 7 – Related Works (Deep Reinforcement Learning)
This table presents Deep Reinforcement Learning applications in wireless sensor networks.  
DRL methods show promise for adaptive routing and resource allocation in other WSN sectors.  
These successful applications motivated us to explore DRL for lifetime maximization, though scalability remains a challenge.  

---

## Slide 8 – Research Gap & Motivations
From Related works, we found gaps in scalability, weak adaptation, and narrow reward focus.  
Our motivation is to create one scalable DRL framework that balances energy use and extends network lifetime.  

---

## Slide 9 – Problem Statement
WSNs face energy holes near sink nodes, static routing limitations, and performance degradation in large networks.  
Our solution uses DRL-powered intelligent routing, adaptive power management, and multi-objective optimization.  

---

## Slide 10 – Research Challenges
Main challenges include scalability of state-action space, energy overhead of DRL computation, dynamic environment changes, partial observability, and learning stability issues.  

---

## Slide 11 – Our Contributions
We made three key contributions:  
First, DRL-based power-saving routing framework with notable lifetime improvements.  
Second, comprehensive validation on 10 to 100 node networks.  
Third, establishing DRL as scalable solution for modern WSN applications.  

---

## Slide 12 – Preliminaries
This section explains basic concepts before our methodology.  

---

## Slide 13 – Deep Reinforcement Learning
Deep Reinforcement Learning (DRL) is a revolutionary Artificial Intelligence methodology that combines reinforcement learning and deep neural networks.
Reinforcement learning means the agent learns by trial and error — it sees a state, takes an action, and then gets a reward or penalty.
The neural network helps the agent handle many inputs and choose the best action.
To make learning stable, DRL uses experience replay, where past experiences are stored and reused.
It also uses two networks — a policy network to select actions and a target network to give stable training feedback.
In this way, DRL gradually learns an energy-efficient policy for complex environments.
For WSNs, DRL adapts to dynamic conditions and learns energy-efficient routing decisions.  

---

## Slide 14 – System Model

This figure shows the overall system model of PSR-DRL.

At first, the WSN module manages how nodes are deployed and clustered.
Next, the data processing module collects sensor data and only sends important information to reduce load.
The DRL module is the brain — it takes the current network state and selects the best action for routing.
The power-saving module decides sleep schedules for nodes.
Finally, the energy management module tracks energy use and gives feedback to DRL.
All these modules together help reduce power consumption and extend network lifetime.

---

## Slide 15 – Deep Q-Learning Architecture
Our PSR-DRL uses Double DQN with policy and target networks.  
Experience replay stores 10,000 interactions for stable learning.  
Neural network has two hidden layers with 64 neurons each.  

---

## Slide 16 – DQL Architecture Figure
This figure shows our Deep Q-Learning architecture for routing in WSN.  
The neural network has two hidden layers with 64 neurons each, which approximate the Q-values for different actions.
Based on the current network state, the policy network selects the best action for routing.
To make learning stable, the target network helps calculate Q-values and prevents overestimation.
Both networks are updated regularly to improve decision accuracy.  
Experience replay stores past actions, states, and rewards in a memory buffer.  
This allows the agent to learn from previous experiences and break the correlation between samples, making training more stable.    
Together, these modules allow the agent to make efficient and adaptive routing decisions, which saves energy in the network.

---

## Slide 17 – Agent Design: State, Actions & Reward
In our design, each node is an agent.
It sees a 9-dimensional state: energy, distance to cluster head, distance to sink, hop count, data urgency, congestion, sleep pressure, cluster health, and time pattern.
It can take 3 actions: send to cluster head, send to sink, or go to sleep mode.The reward function gives positive values for successful delivery, higher energy, longer lifetime, and shorter path.
This helps the agent learn to balance energy use and performance.

---

## Slide 18 – Regular Node Workflow
This figure shows how a regular node works.
First, it checks if the new data is important.
If yes, the DRL agent decides the best action: send to cluster head, send directly to sink, or go to sleep.
This way, useless data is not transmitted, saving energy.

---

## Slide 19 – Cluster Head Workflow
Here is the workflow for cluster heads.
Cluster head collects data from all member nodes, combines it to remove redundancy, and then decides the best path.
It can send data directly to sink or through another cluster head.
Cluster heads are rotated every 300 seconds, so that one node does not lose energy too quickly.

---

## Slide 20 – Simulation Setup
For simulation, we used Python with libraries like NumPy, Pandas, Matplotlib, and PyTorch.
The hardware was Ubuntu 24.04 with Intel Core i5 processor, 16 GB RAM, and an NVIDIA GPU.
This setup was used to train and test our proposed system.

---

## Slide 21 – Evaluation Criteria & Parameters
We measured performance using two metrics:
First Node Death Time — how long before the first node dies, which shows balance in energy.
And Network Lifetime — how long most nodes stay alive in the network.
These two metrics show how energy-efficient and reliable the system is.

We scaled network parameters for different sizes but kept DRL parameters constant.  

---

## Slide 22 – Network Parameters
This table shows simulation parameters for 10, 30, 50, and 100 nodes.  
Parameters include clusters, send range, area size, and energy values scaled appropriately.  

---

## Slide 23 – DQL Parameters
Here are the deep Q-learning parameters.
We used same learning rate, batch size, discount factor, epsilon, and memory for all network sizes.
This means our system does not need to be tuned separately for small or large networks.  

---

## Slide 24 – First Node Death Comparison Table
This table summarizes all first node death results.
On average, PSR-DRL improved 84% over RLBEEP, and 449% over EER-RL.
So, our method clearly provides balanced energy use and stability.

---

## Slide 25 – 10-Node First Node Death
This graph shows First Node Death for 10 nodes.
Our PSR-DRL curve stays alive much longer than RLBEEP and EER-RL.
45.6% improvement over RLBEEP, 434.5% over EER-RL.  
It clearly proves better energy balance, even in small networks. 

---

## Slide 26 – 30-Node First Node Death
Here is the result for 30 nodes.
Again, PSR-DRL performs better, delaying first node death by around 20% over RLBEEP and more than 3 times over EER-RL.
So, our method works well as the network grows.
 

---

## Slide 27 – 50-Node First Node Death
This figure shows 50 nodes case.
Our system delays node death almost 60% better than RLBEEP and 4 times better than EER-RL.
This shows the scalability of PSR-DRL. 

---

## Slide 28 – 100-Node First Node Death
PNow for the 100-node network.
The difference is very big here.
PSR-DRL delays the first node death by more than 200% compared to RLBEEP, and 600% compared to EER-RL.
This proves our method is highly scalable.

---

## Slide 29 – Network Lifetime Comparison Table
This table summarizes the network lifetime results.
Our methods outperforms both methods across all network sizes.  
On average, PSR-DRL improved 28% over RLBEEP, and 574% over EER-RL.
So, our method is clearly more reliable and energy-efficient.

---

## Slide 30 – 10-Node Network Lifetime
This graph shows lifetime for 10 nodes.
Our method maintains all nodes alive for a much longer time than others.
EER-RL fails very quickly, but PSR-DRL is stable.

---

## Slide 31 – 30-Node Network Lifetime
Here is the result for 30 nodes.
PSR-DRL again outperforms, but RLBEEP is closer in this case.
Still, EER-RL fails too early.

---

## Slide 32 – 50-Node Network Lifetime
This figure shows 50 nodes.
PSR-DRL maintains all nodes alive for over 2300 seconds, much longer than RLBEEP and far better than EER-RL.
This shows strong energy efficiency.  
Consistent performance difference proves reliability.  

---

## Slide 33 – 100-Node Network Lifetime
Finally, for 100 nodes.
PSR-DRL extended lifetime to 3533 seconds, which is about 30% better than RLBEEP and more than double compared to EER-RL.
This proves our method is very effective in large-scale networks.

---

## Slide 34 – Conclusions
PSR-DRL achieved significant improvements in energy efficiency and network lifetime.  
First Node Death: 84.2% over RLBEEP, 449.3% over EER-RL.  
Network Lifetime: 36.1% over RLBEEP, 574.0% over EER-RL.  
Suitable for environmental monitoring, agriculture, and smart city applications.  

---

## Slide 35 – Future Works
Future directions include battery life prediction in routing decisions.  
We will explore Graph Neural Networks for smarter routing.  
These will enhance energy savings in large-scale dynamic networks.  

---

## Slide 36-39 – References
These slides list all references from traditional routing methods to deep reinforcement learning approaches.  
We analyzed existing works to identify gaps and build our improved system.  

---

## Slide 40 – Thank You / Q&A
Thank you for your attention.  
This completes our presentation on PSR-DRL.  
Now we are ready to answer your questions.