# Run the Code with Native Distributed Tensorflow and Horovod
First follow the distributed Tensorflow convention to run `experiment.py` as actors,
then follow the Horovod convention to run `experiment.py` as learner(s) with 
`mpirun` (see [Horovod docs](https://github.com/uber/horovod#usage)). 

Example 1: 1 learner, 2 actors.
The learner `localhost:8001` links to the two actors `localhost:9000,localhost:9001`.
Run the following commands in 3 separate terminals.
```bash
mpirun -np 1 \
    -H localhost:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^tcp \
    python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9000,localhost:9001  \
    --job_name=learner --task=0  \
    --level_name=PongNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9000,localhost:9001 \
    --job_name=actor --task=0 \
    --level_name=PongNoFrameskip-v4

python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9000,localhost:9001   \
    --job_name=actor --task=1 \
    --level_name=PongNoFrameskip-v4
```

Example 2:  2 learners 3 actors.
The learner `localhost:8001` links to the two actors `localhost:9000,localhost:9003`,
and the learner `localhost:8002` links to the actor `localhost:9002`.
The two learners `localhost:8001,localhost:8002` are paralleled by `horovod` using 
`mpirun`.
Run the following commands in 4 separate terminals.
```bash
mpirun -H localhost,localhost \
    -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^tcp \
    python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9001,localhost:9003  \
    --job_name=learner --task=0  \
    --level_name=PongNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric \
    : \
    -np 1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^tcp \
    python experiment.py \
    --learner_host=localhost:8002 \
    --actor_hosts=localhost:9002  \
    --job_name=learner --task=0  \
    --level_name=PongNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9001,localhost:9003 \
    --job_name=actor --task=0 \
    --level_name=PongNoFrameskip-v4
python experiment.py \
    --learner_host=localhost:8001 \
    --actor_hosts=localhost:9001,localhost:9003 \
    --job_name=actor --task=1 \
    --level_name=PongNoFrameskip-v4

python experiment.py \
    --learner_host=localhost:8002 \
    --actor_hosts=localhost:9002  \
    --job_name=actor --task=0 \
    --level_name=PongNoFrameskip-v4
```