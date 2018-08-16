# Run the Code with Native Distributed Tensorflow

Follow the distributed Tensorflow convention and run the `experiment.py` 
multiple times by telling whether it is a Parameter Sever, Learner or Actor. 

Example 1. Run the following commands in 3 separate terminals.
```bash
# 1 learner 1 actor
python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001 \
    --actor_hosts=localhost:9000  \
    --job_name=ps --task=0

python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001 \
    --actor_hosts=localhost:9000  \
    --job_name=learner --task=0  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001 \
    --actor_hosts=localhost:9000  \
    --job_name=actor --task=0 \
    --level_name=BreakoutNoFrameskip-v4
```

Example 2. Run the following commands in 6 separate terminals.
```bash
# 2 learners 3 actors
python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=ps --task=0

python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=learner --task=0  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric
python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=learner --task=1  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=actor --task=0 \
    --level_name=BreakoutNoFrameskip-v4
python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=actor --task=1 \
    --level_name=BreakoutNoFrameskip-v4
python experiment.py \
    --ps_hosts=localhost:8000 \
    --learner_hosts=localhost:8001,localhost:8002 \
    --actor_hosts=localhost:9000,localhost:9001,localhost:9002  \
    --job_name=actor --task=2 \
    --level_name=BreakoutNoFrameskip-v4
```