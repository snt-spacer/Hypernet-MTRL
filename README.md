![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Hypernetworks for Control of Reinforcement Learning Autonomous Agents

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


We introduce a novel hypernetwork-based framework for Multi-Task Reinforcement Learning (MTRL) that addresses these limitations by learning a single, generalizable policy for multiple diverse robotic systems:
- **High Speed Racing**: Our framework enables a single policy to successfully race on a variety of unseen tracks.
- **Floating Platform**: a single hypernetwork policy effectively performs four distinct control objectives: stabilization, docking, velocity tracking, and rendezvous.
- **Sim-to-real**: validation for the floating platform tasks.
- **Code and Weights**: Opens-source of the entire stack.


## Getting Started

### Clone the repo and go to the corresponding branch:
```
git clone 
cd Isaaclab_RANS
```
Current Branches:
- [Racing]()
- [Floating Platform]()

### Build and start the docker
```
./docker/container.py build
./docker/container.py start
```

### Train

Racing
```
./scripts/reinforcement_learning/rsl_rl/racing_train.sh
```

Floating Platform
```
./scripts/reinforcement_learning/rsl_rl/control_train_hypernet.sh
```

### Eval
Racing
```
./scripts/reinforcement_learning/rsl_rl/racing_eval.sh
```

Floating Platform
```
./scripts/reinforcement_learning/rsl_rl/control_eval_hypernet.sh
```

### Performance metrics
```
./isaaclab.sh -p source\isaaclab_tasks\isaaclab_tasks\rans\utils\plots\eval_plots_multitask.py
```
