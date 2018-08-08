# 1 learner 1 actor
python experiment.py --num_learners=1 --num_actors=1 \
    --job_name=ps --task=0

python experiment.py --num_learners=1 --num_actors=1 \
    --job_name=learner --task=0  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py --num_learners=1 --num_actors=1 \
    --job_name=actor --task=0 \
    --level_name=BreakoutNoFrameskip-v4


# 2 learners 3 actors
python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=ps --task=0

python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=learner --task=0  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric
python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=learner --task=1  \
    --level_name=BreakoutNoFrameskip-v4 \
    --batch_size=4 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric

python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=actor --task=0 \
    --level_name=BreakoutNoFrameskip-v4
python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=actor --task=1 \
    --level_name=BreakoutNoFrameskip-v4
python experiment.py --num_learners=2 --num_actors=3 \
    --job_name=actor --task=2 \
    --level_name=BreakoutNoFrameskip-v4