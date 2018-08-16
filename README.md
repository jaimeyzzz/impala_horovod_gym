# Our Tweak of IMPALA
Our tweak of the [IMPALA](https://github.com/deepmind/scalable_agent) code.
We modify the original code to support:
* Multiple-machine-multiple-gpu training (with distributed Tensorflow)
* OpenAI Gym compatible
* More Neural Network architectures 

## Dependencies
Install the following python packages:
* `tensorflow`
* `dm-sonnet`
* [`gym`](https://github.com/openai/gym#atari) (with Atari installed)
* `paramiko`
* `libtmux`
* `opencv-python`
Note: the original IMPALA code is with python 2.x,
so we recommend you make a virtual environment of python 2.x and pip install the
above packages.

## Running the Code for Training

### With Native Distributed Tensorflow
Run the `experiment.py` multiple times by telling whether it is a Parameter
Sever, Learner or Actor. Examples:
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

### With Frontend Code
Run the "frontend script" `run_exeriment_mm_raw.py`,
which wraps `experiment.py` by reading the `ps_hosts`, `learner_hosts` and 
`actor_hosts` from a separate csv file prepared beforehand.
Examples:
```bash
python run_exeriment_mm_raw.py \
    --cluster_csv_path=local_cluster_example.csv \
    --agent_name=ResNetLSTMAgent

```

### With Cluster Management Tool
TODO
