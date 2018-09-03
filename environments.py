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

"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf
import gym

from atari_env import make_final

nest = tf.contrib.framework.nest


def my_make_final(level_name):
  return make_final(level_name, episode_life=True, clip_rewards=False,
                    frame_stack=True, scale=False)


def get_action_set(level_name):
  dummy_env = my_make_final(level_name)
  return [i for i in range(dummy_env.action_space.n)]


def get_observation_shape(level_name):
  dummy_env = my_make_final(level_name)
  return dummy_env.observation_space.shape


class PyProcessGym(object):
  """gym wrapper for PyProcess."""
  def __init__(self, level):
    self._env = my_make_final(level)
    assert self._env

  def _reset(self):
    return self._env.reset()
  
  def _trans(self, obs):
    return obs.swapaxes(2, 0)

  def initial(self):
    return self._trans(self._reset())

  def step(self, action):
    observation, reward, done, info = self._env.step(action)
    if done:
        observation = self._reset()
    return (np.float32(reward), done, self._trans(observation),
            np.float32(info['sum_raw_reward']),
            np.int32(info['sum_raw_step']))

  def close(self):
    self._env.close()

  @staticmethod
  def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
    """Returns a nest of `TensorSpec` with the method's output specification."""
    level_name = constructor_kwargs['level']
    observation_spec = tf.contrib.framework.TensorSpec(
      get_observation_shape(level_name),
      tf.uint8
    )

    if method_name == 'initial':
      return observation_spec
    elif method_name == 'step':
      return (
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.bool),
          observation_spec,
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.int32),
      )

StepOutputInfo = collections.namedtuple(
  'StepOutputInfo',
  'episode_return episode_step episode_raw_return episode_raw_step'
)
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation, sum_raw_reward,
        sum_raw_step). `observation` should be the observation after the step is
         taken. If `done` is True, the observation should be the first
        observation in the next episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_reward = tf.constant(0.)
      initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0),
                                    tf.constant(0.), tf.constant(0))
      initial_done = tf.constant(True)
      initial_observation = self._env.initial()

      initial_output = StepOutput(
          initial_reward,
          initial_info,
          initial_done,
          initial_observation)

      # Control dependency to make sure the next step can't be taken before the
      # initial output has been read from the environment.
      with tf.control_dependencies(nest.flatten(initial_output)):
        initial_flow = tf.constant(0, dtype=tf.int64)
      initial_state = (initial_flow, initial_info)
      return initial_output, initial_state

  def step(self, action, state):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      flow, info = nest.map_structure(tf.convert_to_tensor, state)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([flow]):
        reward, done, observation, sum_raw_reward, sum_raw_step = \
          self._env.step(action)

      with tf.control_dependencies(nest.flatten(observation)):
        new_flow = tf.add(flow, 1)

      # When done, include the reward in the output info but not in the
      # state for the next step.
      new_info = StepOutputInfo(info.episode_return + reward,
                                info.episode_step + 1,
                                episode_raw_return=sum_raw_reward,
                                episode_raw_step=sum_raw_step)
      new_state = new_flow, nest.map_structure(
          lambda a, b: tf.where(done, a, b),
          StepOutputInfo(tf.constant(0.), tf.constant(0),
                         new_info.episode_raw_return,
                         new_info.episode_raw_step),
          new_info)
      output = StepOutput(reward, new_info, done, observation)
      return output, new_state
