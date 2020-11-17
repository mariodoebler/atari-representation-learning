import cv2
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym import spaces

from a2c_ppo_acktr.envs import TimeLimitMask, TransposeImage, VecPyTorch, VecNormalize, \
    VecPyTorchFrameStack
from pathlib import Path
import os
import gym
import numpy as np
import torch
from baselines import bench
from test_atariari.wrapper.atari_wrapper import make_atari, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
    ClipRewardEnv, MaxAndSkipEnv, SkipEnv, SkipAndFrameStack, wrap_deepmind
# from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
#     ClipRewardEnv, FrameStack, MaxAndSkipEnv, SkipEnv, SkipAndFrameStack
from .wrapper import AtariARIWrapper
import errno

def make_env(env_id, seed, rank, log_dir, downsample=True, color=False, frame_stack=4):
    def _thunk():
        env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
            env = AtariARIWrapper(env)

        env.seed(seed + rank)


        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=False)

        # if is_atari:
        #     if len(env.observation_space.shape) == 3:
        env = wrap_deepmind(env, downsample=downsample, color=color, frame_stack=frame_stack)
        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        # obs_shape = env.observation_space.shape
        # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = ImageToPyTorch(env)

        return env

    return _thunk

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]))

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))
        # return np.swapaxes(observation, 2, 0)



def make_vec_envs(env_name, seed,  num_processes, num_frame_stack=1, downsample=True, color=False, gamma=0.99, log_dir='./tmp/', device=torch.device('cpu')):
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    envs = [make_env(env_name, seed, i, log_dir, downsample, color, frame_stack=num_frame_stack)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    # if num_frame_stack > 1:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    return envs


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert observations to grayscale."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame


# def wrap_deepmind(env, downsample=True, episode_life=True, clip_rewards=True, frame_stack=False, scale=False,
#                   color=False):
#     """Configure environment for DeepMind-style Atari.
#     """
#     if ("videopinball" in str(env.spec.id).lower()) or ('tennis' in str(env.spec.id).lower()) or ('skiing' in str(env.spec.id).lower()):
#         env = WarpFrame(env, width=160, height=210, grayscale=False)
#     if episode_life:
#         env = EpisodicLifeEnv(env)
#     if 'FIRE' in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     if downsample:
#         env = WarpFrame(env, grayscale=False)
#     if not color:
#         env = GrayscaleWrapper(env)
#     if scale:
#         env = ScaledFloatFrame(env)
#     if clip_rewards:
#         env = ClipRewardEnv(env)
#     # if frame_stack: # before that: observation_sace (210, 160, 1)
#     # env = FrameStack(env, 4) # now: (210, 160, 4)
#     if frame_stack:
#         env = SkipAndFrameStack(env, skip=8, k=4)
#     else:
#         env = SkipEnv(env, skip=4) # doesn't change observation space
#     return env
