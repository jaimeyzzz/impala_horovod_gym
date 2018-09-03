# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

import numpy as np
from six.moves import range
import tensorflow as tf
try:
  import horovod.tensorflow as hvd
  has_horovod = True
except ImportError:
  has_horovod = False  # Actor does not need horovod

import environments
from agent import agent_factory
import py_process
import vtrace


nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '/tmp/agent/', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_string('learner_host', 'localhost:8001',
                    'learner host, only one allowed.')
flags.DEFINE_string('actor_hosts', 'localhost:9001,localhosts:9002',
                    'actor hosts.')
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['ps', 'learner', 'actor'],
                  'Job name.')


# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_string('agent_name', 'SimpleConvNetAgent', 'agent name.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('gradients_clipping', -1.0,
                   'Gradients clipping. Negative number means not clipping. ')

# Environment settings.
flags.DEFINE_string('level_name', 'BreakoutNoFrameskip-v4',
                    '''Level name or gym env name''')
flags.DEFINE_integer('width', 84, 'Width of observation.')
flags.DEFINE_integer('height', 84, 'Height of observation.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')


# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs')
ActorOutputNoState = collections.namedtuple(
    'ActorOutputNoState', 'level_name env_outputs agent_outputs')


def build_actor(agent, env, level_name, action_set):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  initial_agent_state = agent.initial_state(1)
  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output, _ = agent(
      (initial_action,
       nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
      initial_agent_state)
  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state, (initial_env_state, initial_env_output, initial_agent_state,
                     initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_state, agent_output = input_

    # Run agent.
    action = agent_output[0]
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output, agent_state = agent((action, batched_env_output), agent_state)

    # Convert action index to the native action.
    action = agent_output[0][0]
    raw_action = tf.gather(action_set, action)

    env_output, env_state = env.step(raw_action, env_state)

    return env_state, env_output, agent_state, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  _, first_env_output, first_agent_state, first_agent_output = first_values

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, _, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    if hasattr(initial_agent_state, 'c') and hasattr(initial_agent_state, 'h'):
      output = ActorOutput(level_name=level_name, agent_state=first_agent_state,
                           env_outputs=full_env_outputs,
                           agent_outputs=full_agent_outputs)
    else:
      output = ActorOutputNoState(level_name=level_name,
                                  env_outputs=full_env_outputs,
                                  agent_outputs=full_agent_outputs)

    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)


def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)


def build_learner(agent, agent_state, env_outputs, agent_outputs, g_step):
  """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].
    g_step: global step for distributed tf

  Returns:
    Output: A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
    Optimizer: the optimizer
  """
  learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                    agent_state)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, the environment outputs at time step `t` are the inputs that
  # lead to the learner_outputs at time step `t`. After the following shifting,
  # the actions in agent_outputs and learner_outputs at time step `t` is what
  # leads to the environment outputs at time step `t`.
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)
  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

  if FLAGS.reward_clipping == 'abs_one':
    clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  elif FLAGS.reward_clipping == 'soft_asymmetric':
    squeezed = tf.tanh(rewards / 5.0)
    # Negative rewards are given less weight than positive rewards.
    clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.to_float(~done) * FLAGS.discounting

  # Compute V-trace returns and weights.
  # Note, this is put on the CPU because it's faster than on GPU. It can be
  # improved further with XLA-compilation or with a custom TensorFlow operation.
  with tf.device('/cpu'):
    vtrace_returns = vtrace.from_logits(
        behaviour_policy_logits=agent_outputs.policy_logits,
        target_policy_logits=learner_outputs.policy_logits,
        actions=agent_outputs.action,
        discounts=discounts,
        rewards=clipped_rewards,
        values=learner_outputs.baseline,
        bootstrap_value=bootstrap_value)

  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  total_loss = compute_policy_gradient_loss(
      learner_outputs.policy_logits, agent_outputs.action,
      vtrace_returns.pg_advantages)
  total_loss += FLAGS.baseline_cost * compute_baseline_loss(
      vtrace_returns.vs - learner_outputs.baseline)
  total_loss += FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits)

  # Optimization
  num_env_frames = tf.train.get_global_step()
  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)
  optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                        FLAGS.momentum, FLAGS.epsilon)
  # horovod all-reduce optimizer
  optimizer = hvd.DistributedOptimizer(optimizer)
  if FLAGS.gradients_clipping > 0.0:
    grads_and_vars = optimizer.compute_gradients(total_loss)
    grads, vars = zip(*grads_and_vars)
    cgrads, _ = tf.clip_by_global_norm(grads, FLAGS.gradients_clipping)
    grads_and_vars = zip(cgrads, vars)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=g_step)
  else:
    train_op = optimizer.minimize(total_loss, global_step=g_step)

  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(
        FLAGS.batch_size * FLAGS.unroll_length)

  # Adding a few summaries.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)

  return (done, infos, num_env_frames_and_train), optimizer


def create_environment(level_name, seed, is_test=False):
  """Creates an environment wrapped in a `FlowEnvironment`."""

  # Note, you may want to use a level cache to speed of compilation of
  # environment maps. See the documentation for the Python interface of DeepMind
  # Lab.
  config = {
      'width': FLAGS.width,
      'height': FLAGS.height
  }
  if is_test:
    config['allowHoldOutLevels'] = 'true'
    # Mixer seed for evalution, see
    # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
    config['mixerSeed'] = 0x600D5EED
  p = py_process.PyProcess(environments.PyProcessGym, level_name, config)
  return environments.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""
  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs


def train(action_set, level_names):
  """Train."""

  local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
  shared_job_device = '/job:learner/task:0'
  is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
  is_learner = FLAGS.job_name == 'learner'
  actor_hosts = FLAGS.actor_hosts.split(',')
  num_actors = len(actor_hosts)
  learner_host = FLAGS.learner_host.split(',')
  assert(len(learner_host) == 1)
  if is_learner:
    assert(FLAGS.task == 0)
    assert(has_horovod == True)
    hvd.init()

  # Placing the variable on CPU, makes it cheaper to send it to all the
  # actors. Continual copying the variables from the GPU is slow.
  global_variable_device = '/job:learner/task:0' + '/cpu'
  filters = [shared_job_device, local_job_device]
  cluster = tf.train.ClusterSpec({'actor': actor_hosts,
                                  'learner': learner_host})
  config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
  if is_learner:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
  server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                           task_index=FLAGS.task, config=config)

  # Only used to find the actor output structure.
  Agent = agent_factory(FLAGS.agent_name)
  with tf.Graph().as_default():
    agent = Agent(len(action_set))
    env = create_environment(level_names[0], seed=1)
    structure = build_actor(agent, env, level_names[0], action_set)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]
    shapes = [t.shape.as_list() for t in flattened_structure]

  # build graph for actor or learner
  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/cpu'), \
       pin_global_variables(global_variable_device):
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

    # Create Queue and Agent on the learner.
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
      agent = Agent(len(action_set))

    # Build actors and ops to enqueue their output.
    enqueue_ops = []
    for i in range(num_actors):
      if is_actor_fn(i):
        level_name = level_names[i % len(level_names)]
        tf.logging.info('Creating actor %d with level %s', i, level_name)
        env = create_environment(level_name, seed=i + 1)
        actor_output = build_actor(agent, env, level_name, action_set)
        with tf.device(shared_job_device):
          enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

    # Build learner.
    if is_learner:
      # Create global step, which is the number of environment frames
      # processed.
      g_step = tf.get_variable(
        'num_environment_frames', initializer=tf.zeros_initializer(),
        shape=[], dtype=tf.int64, trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
      )
      # Create batch (time major) and recreate structure.
      dequeued = queue.dequeue_many(FLAGS.batch_size)
      dequeued = nest.pack_sequence_as(structure, dequeued)
      def make_time_major(s):
        return nest.map_structure(
          lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]),
          s
        )
      dequeued = dequeued._replace(
          env_outputs=make_time_major(dequeued.env_outputs),
          agent_outputs=make_time_major(dequeued.agent_outputs))

      with tf.device("/gpu"): 
        # Using StagingArea allows us to prepare the next batch and send it to
        # the GPU while we're performing a training step. This adds up to 1
        # step policy lag.
        flattened_output = nest.flatten(dequeued)
        area = tf.contrib.staging.StagingArea(
            [t.dtype for t in flattened_output],
            [t.shape for t in flattened_output])
        stage_op = area.put(flattened_output)
        data_from_actors = nest.pack_sequence_as(structure, area.get())
        # Unroll agent on sequence, create losses and update ops.
        if hasattr(data_from_actors, 'agent_state'):
          agent_state = data_from_actors.agent_state
        else:
          agent_state = agent.initial_state(1)
        output, optimizer = build_learner(
          agent,
          agent_state=agent_state,
          env_outputs=data_from_actors.env_outputs,
          agent_outputs=data_from_actors.agent_outputs,
          g_step=g_step)

    # Create MonitoredSession (to run the graph, checkpoint and log).
    is_chief = is_learner # MonitoredTrainingSession inits all global variables
    hooks = [py_process.PyProcessHook()]
    if is_learner:
      # for variable initialization across learners
      hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_chief)
    if is_learner:
      tf.logging.info('At rank %d', hvd.rank())
    # rank 0 takes care of ckpt saving
    checkpoint_dir = FLAGS.logdir if is_learner and hvd.rank() == 0 else None
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_chief,
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config,
        hooks=hooks) as session:

      if is_learner:
        # tb Logging
        summary_writer = (tf.summary.FileWriterCache.get(FLAGS.logdir) if
                          hvd.rank() == 0 else None)

        # Prepare data for first run.
        session.run_step_fn(
            lambda step_context: step_context.session.run(stage_op))

        # Execute learning and track performance.
        num_env_frames_v = 0
        while num_env_frames_v < FLAGS.total_environment_frames:
          level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
              (data_from_actors.level_name,) + output + (stage_op,))
          level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

          for (level_name, episode_return, episode_step, episode_raw_return,
               episode_raw_step) in zip(
              level_names_v[done_v],
              infos_v.episode_return[done_v],
              infos_v.episode_step[done_v],
              infos_v.episode_raw_return[done_v],
              infos_v.episode_raw_step[done_v]):
            episode_frames = episode_step

            tf.logging.info(
              'learner rank: %d, Env: %s Episode return: %f '
              'Episode raw return: %f',
              hvd.rank(), level_name, episode_return, episode_raw_return
            )

            if hvd.rank() == 0:  # tb Logging
              summary = tf.summary.Summary()
              summary.value.add(tag=level_name + '/episode_return',
                                simple_value=episode_return)
              summary.value.add(tag=level_name + '/episode_frames',
                                simple_value=episode_frames)
              summary.value.add(tag=level_name + '/episode_raw_return',
                                simple_value=episode_raw_return)
              summary.value.add(tag=level_name + '/episode_raw_frames',
                                simple_value=episode_raw_step)
              summary_writer.add_summary(summary, num_env_frames_v)
      else:
        # Execute actors (they just need to enqueue their output).
        while True:
          session.run(enqueue_ops)


def test(action_set, level_names):
  """Test."""

  Agent = agent_factory(FLAGS.agent_name)
  level_returns = {level_name: [] for level_name in level_names}
  with tf.Graph().as_default():
    agent = Agent(len(action_set))
    outputs = {}
    for level_name in level_names:
      env = create_environment(level_name, seed=1, is_test=True)
      outputs[level_name] = build_actor(agent, env, level_name, action_set)

    with tf.train.SingularMonitoredSession(
        checkpoint_dir=FLAGS.logdir,
        hooks=[py_process.PyProcessHook()]) as session:
      for level_name in level_names:
        tf.logging.info('Testing level: %s', level_name)
        while True:
          done_v, infos_v = session.run((
              outputs[level_name].env_outputs.done,
              outputs[level_name].env_outputs.info
          ))
          returns = level_returns[level_name]
          returns.extend(infos_v.episode_return[1:][done_v[1:]])

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  action_set = environments.get_action_set(FLAGS.level_name)
  level_names = [FLAGS.level_name]

  if FLAGS.mode == 'train':
    train(action_set, level_names)
  else:
    test(action_set, level_names)


if __name__ == '__main__':
  tf.app.run()
