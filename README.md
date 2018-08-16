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
Follow the distributed Tensorflow convention and run the `experiment.py` 
multiple times by telling whether it is a Parameter Sever, Learner or Actor. 

See [examples here](sandbox/example_dtf.md).

### With Frontend Code
Run the "frontend script" `run_exeriment_mm_raw.py`,
which wraps `experiment.py` by reading the `ps_hosts`, `learner_hosts` and 
`actor_hosts` from a separate csv file prepared beforehand.
Examples:
```bash
python run_exeriment_mm_raw.py \
    --cluster_csv_path=sandbox/local_cluster_example.csv \
    --agent_name=ResNetLSTMAgent
```

### With Cluster Management Tool
TODO
