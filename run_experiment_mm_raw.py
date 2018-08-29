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

""" frontend script that runs experiment.py in multiple machines with raw ip
specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
from time import sleep
import tempfile
import os
import socket

import paramiko
import tensorflow as tf
from six.moves import shlex_quote
import libtmux

_RUN_WORKER_LOCAL_PRE_CMDS = [
  'echo impala_gym',
]
#_RUN_WORKER_LOCAL_PRE_CMDS = [
#  'conda activate py27',
#  'conda deactivate',
#  'conda activate py27',
#]

_RUN_WORKER_SSH_PRE_CMDS = [
  'pkill python',
  'pkill python',
  'cd impala_gym'
]


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


# Flags used for distributed training.
flags.DEFINE_string(
  'workers_csv_path', 'sandbox/local_workers_example.csv',
  """Workers description file in CSV format. Each row:  """
  """ip, job_name, tf_port, cuda_visible_devices, ssh_port, ssh_username, ssh_password """
)
flags.DEFINE_string('tmux_sess', 'impala',
                    'tmux session name for localhost tasks')

# below flags passed to callee
flags.DEFINE_string('logdir', '/tmp/agent/', 'TensorFlow log directory.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_string('agent_name', 'SimpleConvNetAgent', 'agent name.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')

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


TFClusterDesc = collections.namedtuple(
  'TFClusterDesc',
  'learner_hosts actor_hosts'
)
WorkerDesc = collections.namedtuple(
  'WorkerDesc',
  'job ip tf_port cuda_visible_devices ssh_port ssh_username ssh_password'
)
WorkerSetDesc = collections.namedtuple(
  'WorkerSetDesc',  # each learner or actor is a WorkerDesc. Only ONE learner
  'learner actors'
)


def _to_cmd_str(cmds):
  if isinstance(cmds, (list, tuple)):
    #cmds = " ".join(shlex_quote(str(v)) for v in cmds)
    cmds = ' '.join(str(v) for v in cmds)
  return cmds


def cmd_learners_mpirun_common(worker_sets):
  learners = [ws.learner[0] for ws in worker_sets]
  hosts = ','.join([str(l.ip) for l in learners])
  return [
    #'CUDA_VISIBLE_DEVICES=6,7',
    'mpirun',
    '--allow-run-as-root',
    '-H', hosts,
  ]


def cmd_learner_mpirun_prefix(cluster_desc, worker_desc, task):
  assert (worker_desc.job == 'learner')
  return [
    '-np', str(1),
    '-bind-to', 'none',
    '-map-by', 'slot',
    '-mca', 'pml ob1',
    '-mca', 'btl ^openib',
    '-x', 'NCCL_DEBUG=INFO',
    '-x', 'http_proxy=',
    '-x', 'https_proxy=',
    #'-x', 'CUDA_VISIBLE_DEVICES={}'.format(worker_desc.cuda_visible_devices)
  ]


def cmd_learner(cluster_desc, worker_desc, task):
  assert(worker_desc.job == 'learner')
  return [
    "python",
    "experiment.py",
    "--learner_host={}".format(','.join(cluster_desc.learner_hosts)),
    "--actor_hosts={}".format(','.join(cluster_desc.actor_hosts)),
    "--task={}".format(task),
    "--job_name=learner",
    "--total_environment_frames={}".format(FLAGS.total_environment_frames),
    "--batch_size={}".format(FLAGS.batch_size),
    "--unroll_length={}".format(FLAGS.unroll_length),
    "--num_action_repeats={}".format(FLAGS.num_action_repeats),
    "--seed={}".format(FLAGS.seed),
    "--agent_name={}".format(FLAGS.agent_name),
    "--entropy_cost={}".format(FLAGS.entropy_cost),
    "--baseline_cost={}".format(FLAGS.baseline_cost),
    "--discounting={}".format(FLAGS.discounting),
    "--reward_clipping={}".format(FLAGS.reward_clipping),
    "--level_name={}".format(FLAGS.level_name),
    "--width={}".format(FLAGS.width),
    "--height={}".format(FLAGS.height),
    "--learning_rate={}".format(FLAGS.learning_rate),
    "--decay={}".format(FLAGS.decay),
    "--momentum={}".format(FLAGS.momentum),
    "--epsilon={}".format(FLAGS.epsilon)
  ]


def cmd_actor(cluster_desc, worker_desc, task):
  assert(worker_desc.job == 'actor')
  return [
    "http_proxy=",
    "https_proxy=",
    "CUDA_VISIBLE_DEVICES={}".format(worker_desc.cuda_visible_devices),
    "python",
    "experiment.py",
    "--learner_host={}".format(','.join(cluster_desc.learner_hosts)),
    "--actor_hosts={}".format(','.join(cluster_desc.actor_hosts)),
    "--task={}".format(task),
    "--job_name=actor",
    "--num_action_repeats={}".format(FLAGS.num_action_repeats),
    "--agent_name={}".format(FLAGS.agent_name),
    "--level_name={}".format(FLAGS.level_name),
    "--unroll_length={}".format(FLAGS.unroll_length),
  ]


def parse_workers(workers_csv_path):
  workers = []
  with open(workers_csv_path, "rb") as f:
    reader = csv.DictReader(f, delimiter=",")
    for i, line in enumerate(reader):
      workers.append(WorkerDesc(**line))
  return workers


def split_worker_sets(all_workers):
  learners = [w for w in all_workers if w.job == 'learner']
  actors = [w for w in all_workers if w.job == 'actor']

  worker_sets = []
  for l in learners:
    worker_sets.append(WorkerSetDesc(learner=[l], actors=[]))
  for i, a in enumerate(actors): # "link" to a learner in round-robin order
    i_learner = i % len(worker_sets)
    worker_sets[i_learner].actors.append(a)
  return worker_sets


def to_tf_cluster(worker_set):
  w = worker_set.learner[0]
  item = '{}:{}'.format(w.ip, w.tf_port)
  learner_hosts = [item]

  actor_hosts = []
  for w in worker_set.actors:
    item = '{}:{}'.format(w.ip, w.tf_port)
    actor_hosts.append(item)
  return TFClusterDesc(learner_hosts=learner_hosts, actor_hosts=actor_hosts)


def _long_cmd_to_tmp_file(cmd_str):
  fd, file_path = tempfile.mkstemp(suffix='.sh')
  with os.fdopen(fd, "w") as f:
    f.write(cmd_str)
  return file_path


def run_cmds_local(cmds, tmux_sess_name, tmux_win_name):
  print('sending command to tmux sess {}'.format(tmux_sess_name))
  print("\n".join(cmds))

  # find or create the session
  tmux_server = libtmux.Server()
  tmux_sess = None
  try:
    tmux_sess = tmux_server.find_where({'session_name': tmux_sess_name})
  except:
    pass
  if tmux_sess is None:
    tmux_sess = tmux_server.new_session(tmux_sess_name)
  # create new window/pane, get it and send the command
  tmux_sess.new_window(window_name=tmux_win_name)
  pane = tmux_sess.windows[-1].panes[0]
  # run the command
  pre_cmds = _RUN_WORKER_LOCAL_PRE_CMDS
  cmd_str = "\n".join(pre_cmds + cmds)
  if len(cmd_str) < 512:
    pane.send_keys(cmd_str, suppress_history=False)
    sleep(0.6)
  else:
    # tmux may reject too long command
    # so let's write it to a temp file, and run it in tmux
    cmd_str = "\n".join(cmds)
    tmp_file_path = _long_cmd_to_tmp_file(cmd_str)
    tmp_cmd_str = pre_cmds + ["cat {}".format(tmp_file_path),
                              "sh {}".format(tmp_file_path)]
    pane.send_keys("\n".join(tmp_cmd_str), suppress_history=False)
    sleep(0.7)
    #pos.unlink(tmp_file_path)

  print("done.\n")


def run_cmds_ssh(cmds, ip, port, username, password):
  pre_cmds = _RUN_WORKER_SSH_PRE_CMDS
  cmds = pre_cmds + cmds
  cmd_str = "\n".join(cmds)
  print("sending command to {}@{} -p{} with password {}".format(
    username, ip, port, password))
  print(cmd_str)

  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(ip, port=port, username=username, password=password)
  ssh.exec_command(cmd_str)

  print("done.\n")


def _is_ip_local_machine(ip_str):
  all_local_ips = ['localhost', '127.0.0.1']
  try:
    local_real_ip = socket.gethostbyname(socket.gethostname())
    all_local_ips.append(local_real_ip)
  except Exception:
    pass
  return ip_str in all_local_ips


def run_actor(tf_cluster, worker, task):
  assert(worker.job == 'actor')

  cmds = []
  cmds.append(_to_cmd_str(cmd_actor(tf_cluster, worker, task)))

  if _is_ip_local_machine(worker.ip):
    run_cmds_local(cmds, tmux_sess_name=FLAGS.tmux_sess,
                   tmux_win_name=worker.job)
  else:
    run_cmds_ssh(cmds, ip=worker.ip, port=worker.ssh_port,
                 username=worker.ssh_username,
                 password=worker.ssh_password)


def run_learners(worker_sets):
  cmds_common = cmd_learners_mpirun_common(worker_sets)
  cmds_per = []
  for w_set in worker_sets:
    tf_cluster = to_tf_cluster(w_set)
    learner = w_set.learner[0]  # should be only one
    cmd = cmd_learner_mpirun_prefix(tf_cluster, learner, task=0)
    cmd += cmd_learner(tf_cluster, learner, task=0)
    cmds_per.append(_to_cmd_str(cmd))
  cmds = [_to_cmd_str(cmds_common) + ' ' + ' : '.join(cmds_per)]

  run_cmds_local(cmds, FLAGS.tmux_sess, tmux_win_name='learners')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  all_workers = parse_workers(FLAGS.workers_csv_path)
  worker_sets = split_worker_sets(all_workers)

  # run actors in each worker set
  failed_actors = []
  for w_set in worker_sets:
    tf_cluster = to_tf_cluster(w_set)
    task = 0
    for worker in w_set.actors:
      try:
        run_actor(tf_cluster, worker, task)
      except Exception:
        failed_actors.append(worker)
      task += 1

  # run learners across all worker sets
  run_learners(worker_sets)

  print('failed actors: {}'.format(len(failed_actors)))
  print(failed_actors)


if __name__ == '__main__':
  tf.app.run()
