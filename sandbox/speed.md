## Number of Actors per Learner
We want to see how many actors a learner can "carry". 
We test the code with the environment `PongNoFrameskip-v4`,
with the larget net `ResNetLSTMAgent`, `batch_size = 32`, `unroll_length = 100`.
The learner is a single GPU card, 
while each actor is a remote CPU machine seeing 4 cores (allocated by our private cloud service).
The training throughput is as follows (Note: you should multply the number by 4 when counting the environment frames
due to the repeat-action-4-frames ):
| #Learners | #Actors | speed |
|-----------|---------|-------|
| 1         | 15      | 2.3K  |
| 1         | 25      | 4.3K  |
| 1         | 35      | 6.0K  |
| 1         | 45      | 6.2K  |
| 1         | 50      | 6.2K  |
| 1         | 110     | 6.3K  |

We can see that a learner can carry at approximately 45 actors, 
and adding more actors will not significantly improve the throughput.

## Scale-up when Adding More Learners
Based on the 45-actors-per-learner results, 
we further test the scale-up when adding more learners.
The settings are the same with previous experiment.
Each learner is a single GPU card mounted on a single machine, 
and we test up to 8 learners.
The training throughput is as follows  (Note: you should multply the number by 4 when counting the environment frames
due to the repeat-action-4-frames ):
| #Learners | #Actors | speed |
|-----------|---------|-------|
| 1         | 45      | 5.8K  |
| 2         | 90      | 10.8K |
| 3         | 135     | 15.6K |
| 4         | 180     | 18.8K |
| 5         | 225     | 21.0K |
| 6         | 270     | 21.6K |
| 7         | 315     | 21.7K |
| 8         | 360     | 21.6K |

We can see that the scale-up is good for 1 to 4 learners, 
and becomes bad when more learners are involved.
Note, this result is just meaningful for the specific Neural Network architecture and hyper-parameters.
For example, even bigger neural net would benefit more from multiple-learner settings,
although no public RL literature has disscussed the application of a net bigger than the `ResNetLSTM`.
Note also that the speed may be affected by the unstable network transportation,
as shown in the two tables for the case of 1 learner.
