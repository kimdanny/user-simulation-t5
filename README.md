# user-simulation
User Simulator that generates user-side utterance, predicts user's next action and satisfaction level.

![t5-mtl](imgs/t5-mtl-diagram.png)

We propose a multi-task based deep learning user simulator for goal oriented dialogue system that is trained to predict users’ satisfaction and action, while generating the users’ next utterance at the same time with shared weights. 
We show that 
1) a deep text-to-text multi-task neural model achieves state-of-the-art performance in user satisfaction and action prediction
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

