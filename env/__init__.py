from gym import register

from .config import config


register(
    id='acrobot-v0',
    entry_point='env.acrobot.env:AcrobotEnv',
    max_episode_steps=config['acrobot']['N'],
)

register(
    id='pendulum-v0',
    entry_point='env.pendulum.env:PendulumEnv',
    max_episode_steps=config['pendulum']['N'],
)
