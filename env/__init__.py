from gym import register

register(
    id='acrobot-v0',
    entry_point='env.env:AcrobotEnv',
    max_episode_steps=50,
)