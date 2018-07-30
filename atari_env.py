import gym, random
import numpy as np
from memoire_tf.actor.env.atari_wrappers import make_atari, wrap_deepmind

class TransposeWrapper(gym.ObservationWrapper):
  def observation(self, observation):
    return np.transpose(np.array(observation), axes=(2,0,1))

class NoRwdResetEnv(gym.Wrapper):
  def __init__(self, env, no_reward_thres):
    """Reset the environment if no reward received in N steps
    """
    gym.Wrapper.__init__(self, env)
    self.no_reward_thres = no_reward_thres
    self.no_reward_step = 0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if reward == 0.0:
      self.no_reward_step += 1
    else:
      self.no_reward_step = 0
    if self.no_reward_step > self.no_reward_thres:
      done = True
    return obs, reward, done, info

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self.no_reward_step = 0
    return obs

def make_final(env_id, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
  env = wrap_deepmind(make_atari(env_id), episode_life, clip_rewards, frame_stack, scale)
  env = TransposeWrapper(env)
  env = NoRwdResetEnv(env, no_reward_thres = 1000)
  return env

if __name__ == '__main__':
  #env = make_final('BreakoutNoFrameskip-v4', True, True, False, False)
  #env = make_final('SeaquestNoFrameskip-v4', True, True, False, False)
  env = make_final('QbertNoFrameskip-v4', True, True, False, False)
  #env = make_final('MontezumaRevengeNoFrameskip-v4', True, True, False, False)
  print(env.action_space)

  n_game = 5
  epi_max_len = 4096

  game_idx = 0
  for game_idx in range(n_game):
    # start
    obs = env.reset()
    for i in range(epi_max_len):
      action = random.randint(0,3)
      obs, rwd, term, info = env.step(action) # discrete
      if rwd != 0.0:
        print(rwd)
      if term:
        print(obs.shape)
        print(obs.dtype)
        print(i)
        break
    # close
    print("game_idx: %d" % game_idx)
