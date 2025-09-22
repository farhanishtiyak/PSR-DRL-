# PSR-DRL: Power Saving Wireless Routing with Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/LaTeX-supported-green.svg)](https://www.latex-project.org/)

A comprehensive thesis research project implementing **Deep Reinforcement Learning (DRL)** for energy-efficient routing in Wireless Sensor Networks (WSNs). This work demonstrates how intelligent machine learning approaches can significantly extend network lifetime through adaptive power management and optimal routing decisions.

## 🎯 Research Overview

This thesis develops **PSR-DRL** (Power Saving Wireless Routing based on Deep Reinforcement Learning), addressing the critical challenge of energy optimization in resource-constrained wireless sensor networks. The system uses deep reinforcement learning algorithms to make intelligent routing decisions in real time that keep energy use balanced throughout the network, demonstrating exceptional scalability and performance improvements across diverse network configurations (10, 30, 50, and 100 nodes).

### Key Achievements
- **84.2% improvement** in First Node Death (FND) timing over RLBEEP (averaging across all network sizes)
- **449.3% improvement** in First Node Death over EER-RL across all configurations
- **28.2% improvement** in network lifetime over RLBEEP
- **574.1% improvement** in network lifetime over EER-RL
- Superior scalability with exceptional performance in large-scale deployments

## 🏗️ System Architecture

The PSR-DRL system integrates multiple components:

### Deep Q-Network (DQN) Architecture
- **State Space**: 9-dimensional feature vector (energy, distance, congestion, temporal factors)
- **Action Space**: Intelligent next-hop selection with adaptive exploration
- **Experience Replay**: 10,000-capacity memory buffer with batch training
- **Target Network**: Stable learning with periodic updates every 10 episodes

### Power Saving Framework
- **Adaptive Sleep Scheduling**: Node-specific sleep patterns with 95% energy reduction
- **Intelligent Transmission Control**: Threshold-based decision making
- **Heterogeneous Energy Management**: 0-30% node-specific consumption variation
- **Dynamic Cluster Head Rotation**: Energy-aware leadership changes every 300s

## 📁 Repository Structure

```
PSR-DRL/
├── 📄 main_fixed.tex              # Main thesis LaTeX document
├── 📄 system_model.tex            # System model documentation
├── 📄 drl_based_routing.tex       # DRL routing algorithms
├── 🐍 drl_architecture_design.py  # DRL system architecture implementation
├── 🐍 system_architecture_design.py # System design and visualization
├── 🐍 data_forwarding_flow.py     # Data flow implementation
├── 🐍 update_network_lifetime.py  # Network lifetime analysis
├── 📊 run_epochs.m                # MATLAB simulation scripts
│
├── 📂 Documentation/              # Comprehensive thesis documentation
│   ├── EXECUTIVE_SUMMARY.md      # Research overview and achievements
│   ├── FINAL_RESEARCH_REPORT.md  # Complete research report
│   ├── DRL_DOCUMENTATION.md      # DRL implementation details
│   └── ALGORITHM_FIXED.tex       # Algorithm specifications
│
├── 📂 Dataset/                    # Real WSN sensor data
│   ├── node1.csv - node10.csv    # Temperature, humidity, voltage readings
│   └── README                     # Dataset description
│
├── 📂 All figures/                # Research visualizations
│   ├── PSR_DRL_Architecture.pdf   # System architecture diagrams
│   ├── DRL_Detailed_Architecture.pdf # DRL component details
│   ├── network_lifetime_comparison*.pdf # Performance comparisons
│   └── overlay_comparison_*.pdf   # Method comparison results
│
├── 📂 Comparison/                 # Experimental results
│   └── {10,30,50,100}/           # Results for different network sizes
│
├── 📂 Competitor/                 # Baseline comparison implementations
│   └── compare_simulations.py    # Comparative analysis scripts
│
└── 📂 EER-RL/                    # Energy-efficient routing with RL
    ├── Epoch/                     # Training epoch results
    └── withoutEpoch/             # Non-epochal training results
```

## 🚀 Getting Started

### Prerequisites

**Python Requirements:**
```bash
Python 3.8+
numpy >= 1.21.0
matplotlib >= 3.5.0
torch >= 1.10.0
pandas >= 1.3.0
```

**LaTeX Requirements:**
```bash
LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
Required packages: algorithm, algorithmic, amsmath, amssymb, geometry
```

**MATLAB Requirements:**
```bash
MATLAB R2020a or later
Statistics and Machine Learning Toolbox
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/farhanishtiyak/PSR-DRL-.git
cd PSR-DRL-
```

2. **Install Python dependencies:**
```bash
pip install numpy matplotlib torch pandas scipy
```

3. **Verify LaTeX installation:**
```bash
pdflatex --version
```

## 🎮 Usage

### Running DRL Simulations

1. **Execute the main DRL architecture:**
```bash
python drl_architecture_design.py
```

2. **Run system architecture visualization:**
```bash
python system_architecture_design.py
```

3. **Analyze data forwarding flow:**
```bash
python data_forwarding_flow.py
```

4. **Evaluate network lifetime:**
```bash
python update_network_lifetime.py
```

### MATLAB Simulations

1. **Run epoch-based training:**
```matlab
run_epochs
```

2. **Compare with baseline methods:**
```matlab
cd Competitor
run compare_simulations.py
```

### LaTeX Compilation

1. **Compile main thesis document:**
```bash
pdflatex main_fixed.tex
pdflatex main_fixed.tex  # Run twice for references
```

2. **Compile individual sections:**
```bash
pdflatex system_model.tex
pdflatex drl_based_routing.tex
```

## 📊 Experimental Results

### Network Performance Metrics

#### First Node Death Time Comparison
| Network Size | PSR-DRL (sec) | RLBEEP (sec) | EER-RL (sec) | vs RLBEEP | vs EER-RL |
|-------------|---------------|--------------|-------------|-----------|-----------|
| **10 Nodes** | 1514.1 | 1039.5 | 283.3 | **+45.6%** | **+434.5%** |
| **30 Nodes** | 1722.0 | 1427.3 | 405.1 | **+20.7%** | **+325.1%** |
| **50 Nodes** | 2117.5 | 1343.9 | 412.7 | **+57.6%** | **+413.1%** |
| **100 Nodes** | 2203.5 | 704.5 | 304.1 | **+212.8%** | **+624.6%** |
| **Average** | 1889.3 | 1128.8 | 351.3 | **+84.2%** | **+449.3%** |

#### Network Lifetime Comparison
| Network Size | PSR-DRL (sec) | RLBEEP (sec) | EER-RL (sec) | vs RLBEEP | vs EER-RL |
|-------------|---------------|--------------|-------------|-----------|-----------|
| **10 Nodes** | 2053 | 1298 | 116 | **+58.2%** | **+1669.8%** |
| **30 Nodes** | 2875 | 2999 | 601 | **-4.1%** | **+378.5%** |
| **50 Nodes** | 3071 | 2374 | 1421 | **+29.4%** | **+116.2%** |
| **100 Nodes** | 3533 | 2733 | 1524 | **+29.3%** | **+131.8%** |
| **Average** | 2883 | 2351 | 915.5 | **+28.2%** | **+574.1%** |

### Key Performance Highlights
- **Exceptional Scalability**: Most significant gains observed in large-scale deployments (100 nodes show +212.8% improvement in FND)
- **Consistent First Node Death Superiority**: PSR-DRL outperforms both competitors across all network sizes
- **Outstanding Performance vs EER-RL**: Demonstrates remarkable improvements, particularly in 10-node networks (+1669.8% network lifetime)
- **Energy Efficiency**: Intelligent real-time routing decisions maintain balanced energy consumption
- **Robust Large-Scale Performance**: 100-node networks achieve 3533s lifetime vs RLBEEP's 2733s and EER-RL's 1524s
- **Cross-Network Consistency**: Average improvements of 84.2% (FND) and 28.2% (lifetime) over RLBEEP across all configurations

### Scalability Analysis

The system demonstrates exceptional performance scaling across network configurations:

- **10 nodes**: Strong baseline with 1514.1s FND and 2053s lifetime
- **30 nodes**: Maintained efficiency with 1722.0s FND, slight RLBEEP advantage in lifetime (competitive performance)
- **50 nodes**: Robust scalability with 2117.5s FND and 3071s lifetime showing consistent improvements
- **100 nodes**: Enterprise-level deployment capability with peak performance metrics

## 🔬 Research Contributions

### 1. **Novel DRL Architecture**
- First implementation of DQN for WSN power-aware routing
- Multi-dimensional state space capturing network dynamics
- Adaptive exploration-exploitation balance

### 2. **Comprehensive Power Management**
- Intelligent sleep scheduling with 95% energy reduction
- Heterogeneous node behavior preventing synchronized failures
- Dynamic cluster head rotation based on energy levels

### 3. **Real-World Validation**
- Integration with actual WSN sensor datasets
- Realistic energy consumption models
- Comprehensive performance evaluation framework

### 4. **Scalable Implementation**
- Tested across 10-100 node networks
- Distributed learning architecture
- Practical deployment considerations

## 📈 Performance Analysis

### Energy Consumption Breakdown
```
Active Mode:     0.1 J/s (baseline)
Sleep Mode:      0.005 J/s (95% reduction)
Transmission:    Variable based on distance and data size
Processing:      Minimal overhead (< 2% of total consumption)
```

### Learning Convergence
- **Initial Exploration**: ε = 0.9 (high exploration)
- **Final Exploitation**: ε = 0.05 (optimal decisions)
- **Convergence Time**: ~500 episodes for stable performance
- **Memory Efficiency**: 10,000 experience buffer with batch size 32

## 🎓 Academic Context

**Thesis Information:**
- **Title**: Power Saving Routing for Wireless Sensor Network using Deep Reinforcement Learning
- **Authors**: 
  - Md. Farhan Ishtiyak Sezar
  - Md. Rashedunnabi 
- **Supervisor**: Md. Aminul Islam, Ph.D., Associate Professor
- **Institution**: Department of Computer Science and Engineering, Jagannath University, Dhaka-1100, Bangladesh
- **Research Area**: Wireless Sensor Networks, Machine Learning, Energy Optimization
- **Methodology**: Experimental research with comprehensive simulation-based validation across multiple network configurations
- **Key Contributions**: Novel DRL-based routing protocol with exceptional scalability and energy efficiency

**Related Publications:**
- Conference presentations on DRL applications in WSNs
- Journal submissions on energy-efficient routing protocols
- Workshop presentations on machine learning for IoT networks

## 🤝 Contributing

This thesis research welcomes academic collaboration and technical contributions:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/enhancement`)
3. **Commit changes** (`git commit -am 'Add new enhancement'`)
4. **Push to branch** (`git push origin feature/enhancement`)
5. **Create Pull Request**

### Research Areas for Extension
- Multi-objective optimization with additional QoS metrics
- Federated learning for distributed WSN environments
- Integration with emerging IoT protocols
- Real-time deployment on hardware testbeds

## 📚 References and Citations

Key research foundations:
- Deep Q-Learning (Mnih et al., 2015)
- Wireless Sensor Network Protocols (Akyildiz et al., 2002)
- Energy-Efficient Routing (Heinzelman et al., 2000)
- Reinforcement Learning for Networks (Sutton & Barto, 2018)

**Citation Format:**
```bibtex
@mastersthesis{psr_drl_2025,
  title={Power Saving Wireless Routing with Deep Reinforcement Learning},
  author={Sezar, Md. Farhan Ishtiyak and Rashedunnabi, Md.},
  year={2025},
  school={Jagannath University},
  address={Dhaka, Bangladesh},
  type={Bachelor's Thesis},
  note={Supervisor: Md. Aminul Islam},
  url={https://github.com/farhanishtiyak/PSR-DRL-}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Researchers**: 
- **Md. Farhan Ishtiyak Sezar** 
- **Md. Rashedunnabi** 

**Supervisor**: **Md. Aminul Islam, Ph.D.**  
Associate Professor, Department of Computer Science and Engineering  
Jagannath University, Dhaka-1100, Bangladesh

**GitHub**: [@farhanishtiyak](https://github.com/farhanishtiyak)  
**Repository**: [PSR-DRL-](https://github.com/farhanishtiyak/PSR-DRL-)

---

### 🌟 Acknowledgments

- **Dr. Md. Aminul Islam** for supervision, guidance, and support throughout the research
- **Jagannath University** Department of Computer Science and Engineering for providing research facilities
- Research community for foundational work in DRL and WSNs
- Open-source contributors for tools and libraries used
- Dataset providers for real-world sensor network data

---

**⚡ Ready to explore energy-efficient wireless networking with AI? Start with the [Documentation](Documentation/) folder for comprehensive research details!**