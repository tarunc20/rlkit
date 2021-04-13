import cv2
import gym
import numpy as np
from gym.spaces.box import Box


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        o, r, d, i = self.env.step(original)
        return o, r, d, i

    def reset(self):
        return self.env.reset()


class ImageUnFlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env.max_steps
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )
        self.reward_ctr = 0

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight),
            reward,
            done,
            info,
        )


class ImageTransposeWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(
            0, 255, (self.env.imwidth, self.env.imheight, 3), dtype=np.uint8
        )
        self.reward_ctr = 0

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight).transpose(1, 2, 0),
            reward,
            done,
            info,
        )


class ImageEnvMetaworld(gym.Wrapper):
    def __init__(
        self,
        env,
        imwidth=84,
        imheight=84,
        reward_scale=1.0,
    ):
        gym.Wrapper.__init__(self, env)
        self.max_steps = self.env.max_path_length
        self.imwidth = imwidth
        self.imheight = imheight
        self.observation_space = Box(
            0, 255, (3 * self.imwidth * self.imheight,), dtype=np.uint8
        )
        self.image_shape = (3, self.imwidth, self.imheight)
        self.num_steps = 0
        self.reward_scale = reward_scale

    def _get_image(self):
        # img = self.env.sim.render(
        #     width=self.imwidth,
        #     height=self.imheight,
        # )

        # use this if using dm control backend!
        img = self.env.render(
            mode="rgb_array", width=self.imwidth, height=self.imheight
        )

        img = img.transpose(2, 0, 1).flatten()
        return img

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="human",
        render_im_shape=(1000, 1000),
    ):
        o, r, d, i = self.env.step(
            action,
            # render_every_step=render_every_step,
            # render_mode=render_mode,
            # render_im_shape=render_im_shape,
        )
        self.num_steps += 1
        o = self._get_image()
        r = self.reward_scale * r
        new_i = {}
        for k, v in i.items():
            if v is not None:
                new_i[k] = v
        return o, r, d, new_i

    def reset(self):
        super().reset()
        self.num_steps = 0
        return self._get_image()


class DictObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def step(
        self,
        action,
    ):
        o, r, d, i = super().step(action)
        return {"image": o}, r, d, i

    def reset(self):
        return {"image": self.env.reset()}
