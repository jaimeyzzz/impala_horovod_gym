# Our Tweak of IMPALA
Our tweak of the [IMPALA](https://github.com/deepmind/scalable_agent) code.
We modify the original code to support:
* Multiple-machine-multiple-gpu training (with distributed Tensorflow and
 [Horovod](https://github.com/uber/horovod))
* OpenAI Gym compatibility
* More Neural Network architectures 

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

## Running the Code for Training
We offer a couple of ways to run the training code, as described below.

### With Native Distributed Tensorflow and Horovod
Follow the distributed Tensorflow convention to run `experiment.py` as actors,
then follow the Horovod convention to run `experiment.py` as learner(s) with 
`mpirun`. 

See [examples here](sandbox/example_dtf.md).

### With Frontend Code
Run the "frontend script" `run_exeriment_mm_raw.py`,
which wraps `experiment.py` by reading the `learner_hosts` and 
`actor_hosts` from a separate csv file prepared beforehand.
Examples:
```bash
python run_exeriment_mm_raw.py \
    --cluster_csv_path=sandbox/local_cluster_example.csv \
    --agent_name=ResNetLSTMAgent
```

### With Cluster Management Tool
TODO
