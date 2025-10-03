# PIDEA
Parallel Island Differential Evolutionary Algorithm (PIDEA) for multi-satellite mission scheduling


### :bulb: (TAES2025) Multi-Satellite Scheduling for Stereo Tracking of Moving Targets via Parallel Island Differential Evolutionary Algorithm 
### [\[Paper Link\]](https://ieeexplore.ieee.org/document/11192053) Accepted by IEEE Transactions on Aerospace and Electronic Systems

<p align="center">
<img src="PIDEA.png" width="1500px" height="350px" />
</p>

### keyword
Satellite Scheduling, Evolutionary Algorithms, Reinforcement Learning, Attitude Control, Intelligent satellite systems, GPU accelerated applications.

> Abstract: The tracking of moving targets using satellite constellations presents significant challenges due to the need for time-sensitive and effective scheduling solutions.
While static target observation has been extensively studied, stereo tracking of moving targets remains a critical research gap, particularly when considering the complexities introduced by real-time movement and operational constraints.
To address these challenges, this paper proposes a pioneering approach that integrates stereo tracking with a Parallel Island Differential Evolutionary Algorithm (PIDEA) for multi-satellite mission scheduling. The PIDEA harnesses the computational power of Graphics Processing Units (GPUs) and employs island evolution strategies to balance computational efficiency with solution diversity, ensuring timely and effective scheduling in dynamic scenarios.
Additionally, a reinforcement learning-based attitude control system is introduced to enable agile satellites to maintain accurate and stable tracking of moving targets, even under challenging conditions.
To further enhance operational adaptability, we incorporate event-driven mechanisms to dynamically trigger rescheduling when significant changes occur, such as satellite availability or energy constraints.
Extensive experiments conducted in multiple moving target tracking scenarios demonstrate the effectiveness and efficacy of the proposed method. 
The results validate its ability to generate near-optimal scheduling solutions that meet the dual demands of time sensitivity and tracking effectiveness, marking a significant step forward in autonomous satellite mission planning for dynamic environments.


<p align="center">
<img src="flowchart.png" width="700px" height="500px" />
</p>

## Getting started
#### <a id="Step1">Step 1</a>: Modify algorithm parameters in METADATA.
#### <a id="Step2">Step 2</a>: Adjust the inputs of [iea_kernel](task_allocation.py)  and fitness evaluation function base on your own mission requirements.
```.bash
python task_allocation.py
```


### Cite the paper
BibTex
```
@ARTICLE{11192053,
  author={Lu, Wenlong and Liu, Bingyan and Mu, Zhongcheng and Wu, Shufan and Song, Yanjie and Razoumny, Vladimir Yu.},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Multi-Satellite Scheduling for Stereo Tracking of Moving Targets via Parallel Island Differential Evolutionary Algorithm}, 
  year={2025},
  volume={},
  number={},
  pages={1-23},
  keywords={Target tracking;Satellites;Dynamic scheduling;Heuristic algorithms;Attitude control;Scheduling algorithms;Optimization;Evolutionary computation;Aerospace and electronic systems;Aerodynamics;Satellite Scheduling;Evolutionary Algorithms;Reinforcement Learning;Attitude Control;Intelligent systems},
  doi={10.1109/TAES.2025.3617044}}

@ARTICLE{10616229,
  author={Lu, Wenlong and Gao, Weihua and Liu, Bingyan and Niu, Wenlong and Wang, Di and Li, Yun and Peng, Xiaodong and Yang, Zhen},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Reinforcement Learning Driven Time-Sensitive Moving Target Tracking of Intelligent Agile Satellite}, 
  year={2024},
  volume={},
  number={},
  pages={1-18},
  keywords={Satellites;Target tracking;Attitude control;Task analysis;Trajectory;Process control;Job shop scheduling;Attitude control;Satellite applications;Decision-making;Intelligent systems},
  doi={10.1109/TAES.2024.3436061}}
```  
