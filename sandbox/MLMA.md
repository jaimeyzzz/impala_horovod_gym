The Multip-Learner-Multiple-Actor implementation is much like the [OpenAI Rapid](https://blog.openai.com/openai-five/#rapid).

Each leaner links to a couple of actors, where the learner-actor communication relies on 
distributed Tensorflow and the code is as-is of the original impala code. 
The actors are suggested to run on cheap CPU machines.

Across the learners, the NN parameters are updated synchoronously using
[Horovod](https://github.com/uber/horovod)). 
Note: the communication among learners only relies on Horovod, 
NOT relies on distributed Tensorflow.
The learners are suggested to run on GPU machines by letting each learner see only one GPU.
 