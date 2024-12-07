# Detour: Decentralized Multi-Robot Off-Road Navigation via Terrain Diffusion and Curriculum Reinforcement Learning

Haonan Yang, Bolei Chen, Jiaxu Kang, Ping Zhong, Yu Sheng

## 1. Overview

<img src="https://github.com/user-attachments/assets/fdce4862-0742-4ea7-80a6-43a8412157d2"  width="60%" align=“center”/>

The multi-robot navigation system has increasingly shown superior performance and broad application prospects in scenarios such as logistics and rescue. Despite the promising performance achieved by existing work in indoor flat terrains, many challenges still need to be overcome in outdoor off-road settings, such as sensing occlusions caused by complex terrains, dangerous tilts, and dynamic collision avoidance. In this paper, a **De**cen**t**ralized m**u**lti-**r**obot **o**ff-road navigation strategy named **Detour** is proposed to tackle the above issues by using terrain diffusion and **C**urriculum **R**einforcement **L**earning (CRL). In particular, the terrain diffusion technique is designed to imagine possible future terrains based on current visual observations, which not only mitigates the occlusion of sensor data by undulating terrains but also enhances the robot's reactivity to potential collisions. By designing an easy-to-hard curriculum for navigation strategy learning, the robot's navigation ability is steadily enhanced to cope with dynamic scenes with complex terrains. In addition, by fully considering the robot's tilt and collision risks, a reward function is crafted for CRL to address the reward sparsity difficulty. Sufficient comparative and ablation studies demonstrate our Detour's superiority. Surprisingly, we experimentally find that terrain diffusion can drastically reduce navigation time steps while improving navigation success and reducing collisions. Experiments with a real robot in outdoor dynamic scenes further validate Detour's feasibility. The experimental code is available at [Detour](https://github.com/Southyang/Detour).

This repository contains code for Detour.

## 2. A demo video for Detour

This is a demo video of the real robot experiments for the paper titled "Detour: Decentralized Multi-Robot Off-Road Navigation via Terrain Diffusion and Curriculum Reinforcement Learning".
[![Detour's Real Robot Experiments Video](https://res.cloudinary.com/marcomontalbano/image/upload/v1733575898/video_to_markdown/images/youtube--uQZHvykxhrQ-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=uQZHvykxhrQ&ab_channel=Southyang "Detour's Real Robot Experiments Video")

## 3. Installation

Clone
```sh
git clone https://github.com/Southyang/Detour.git
```
Configure the conda environment
```
conda create -n Detour python=3.8
conda activate Detour
```
> pytorch 1.10.0

Compile
```sh
cd Detour
catkin_make
```

## 4. Run

Sumilation enviromnent
```
cd Detour
source ./devel/setup.bash
roslaunch detour train_stage1.launch
```

Navigation script
```
cd Detour/src/detour/scripts
mpiexec -n 6 python train_stage1.py
```

**Experimental diagram**

<img src="https://github.com/user-attachments/assets/31eded0a-cf48-4152-ae47-d5daf885a256" width="80%" ></img>

## 5. Acknowledgement
Code acknowledgements:
 - Detour/src/Husky is modified from [Husky](https://github.com/husky/husky).



