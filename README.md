# user-simulation

Official code repository of the paper [A Multi-Task Based Neural Model to Simulate Users in Goal-Oriented Dialogue Systems](https://www.researchgate.net/publication/360276605_A_Multi-Task_Based_Neural_Model_to_Simulate_Users_in_Goal-Oriented_Dialogue_Systems)

User Simulator that generates user-side utterance, predicts user's next action and satisfaction level.  
SOTA in Satisfaction and Action prediction in [USS dataset](https://arxiv.org/pdf/2105.03748).

<p align="center">
    <img src="imgs/t5-mtl-diagram.png" width="400">
</p>
<p align="center">
    <b>Inference example of the trained T5 model on MultiWOZ 2.1 dataset</b>
</p>


We propose a multi-task based deep learning user simulator for goal oriented dialogue system that is trained to predict users’ satisfaction and action, while generating the users’ next utterance at the same time with shared weights. 
We show that 
1) a deep text-to-text multi-task neural model achieves state-of-the-art (SOTA) performance in user satisfaction and action prediction
2) through ablation analysis, adding utterance generation as an auxiliary task can boost the prediction performance via positive transfer between the tasks. 

## Results
**User Satisfaction Predictions**  
![satisfaction](imgs/satisfaction-table.png)  

**User Action Predictions**  
![action](imgs/action-table.png)  

**User-side Utterance Generation**  
![ug](imgs/ug-score-table.png)  

**Cross-domain Unweigted Average Recall on User Satisfaction Prediction**  
![satisfaction](imgs/cross-domain.png)  



## Environment
The author used AWS EC2 Instance to set up the environment:  

- Instance: `Deep Learning AMI (Amazon Linux 2) Version 57.0`  
- conda_env: `source activate pytorch_p38`
- requirements: `pip install -r requirements.txt`

