# Our Tweak of IMPALA
Our tweak of the Deep Reinforcement Learning algorithm [IMPALA](https://github.com/deepmind/scalable_agent).
We modify the original code to support:
* Distributed Multiple-learner-multiple-actor training (see [here](sandbox/MLMA.md) for a brief description)
* OpenAI Gym compatibility
* More Neural Network architectures 
* Algorithm arguments (including: gradients clipping)


## Dependencies
Install the following python packages:
* `tensorflow`
* [`horovod`](https://github.com/uber/horovod)
* `dm-sonnet`
* [`gym`](https://github.com/openai/gym#atari) (with Atari installed)
* `paramiko`
* `libtmux`
* `opencv-python`

Note: the original IMPALA code is written in python 2.x,
so we recommend you make a virtual environment of python 2.x and pip install the
above packages.
Also, you can simply do everything in docker, see description [here](docker/README.md)

## Running the Code for Training
We offer a couple of ways to run the training code, as described below.

### With Native Distributed Tensorflow and Horovod
First follow the distributed Tensorflow convention to run `experiment.py` as actors,
then follow the Horovod convention to run `experiment.py` as learner(s) with 
`mpirun`. 

See [examples here](sandbox/example_dtf.md).

### With Frontend Code
Run the "frontend script" `run_exeriment_mm_raw.py`,
which wraps `experiment.py` by reading the `learner_hosts` and 
`actor_hosts` from a separate csv file prepared beforehand.
Examples:
```bash
python run_experiment_mm_raw.py \
  --workers_csv_path=sandbox/local_workers_example.csv \
  --level_name=BreakoutNoFrameskip-v4 \
  --agent_name=SimpleConvNetAgent \
  --num_action_repeats=1 \
  --batch_size=32 \
  --unroll_length=20 \
  --entropy_cost=0.01 \
  --learning_rate=0.0006 \
  --total_environment_frames=200000000 \
  --reward_clipping=abs_one \
  --gradients_clipping=40.0
```

See `sandbox/local_workers_example.csv` for the CSV fields you must provide.
The field names should be self-explanatory.

If you have access to some cloud service where you can apply many cheap CPU machines,
(Or you happen to be from internal Tencent and have access to the c.oa.com "compute sharing platform",)
see the description [here](sandbox/coa.md) for how to prepare CSV file.

### With Cluster Management Tool
TODO

## Running the Code for Evaluating
TODO

## Case Studies
TODO: figures/tables for the training speed and socres over PongNoFrameskip-v4. 
