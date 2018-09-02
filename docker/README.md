For Learner (GPU machine, tensorflow-gpu, Horovod), use `DockfileLearner`.

For Actor (CPU machine, tensorflow), it does not need GPU and Horovod, so you can use `DockerfileActor`. 

The commands:
```bash
# for learner
docker build -t impala_horovod_gym_learner -f docker/DockerfileLearner .

# for actor
docker build -t impala_horovod_gym_actor -f docker/DockerfileActor .
```