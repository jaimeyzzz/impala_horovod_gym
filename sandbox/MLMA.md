The Multip-Learner-Multiple-Actor implementation is much like the [OpenAI Rapid](https://blog.openai.com/openai-five/#rapid).

Each leaner links to a couple of actors, where the learner-actor communication relies on 
distributed Tensorflow and the code is as-is of the original impala code. 


Across the learners, the NN parameters are updated synchoronously using
[Horovod](https://github.com/uber/horovod)). 
The communication among learners only relies on Horovod, 
NOT relies on distributed Tensorflow.

Note: The actors and the learners can run in either a single machine or multiple machines,
where each learner sees only one GPU card.
When you have hundreds or thousands of actors,
it is suggested that the actors run on cheap CPU machines,
while the learners run on GPU machines.
 