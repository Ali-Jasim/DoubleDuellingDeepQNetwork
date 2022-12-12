import gym

env = gym.make("LunarLander-v2")
obs, info = env.reset()


def policy(obs):
    return env.action_space.sample()


epoch = 100

for i in range(epoch):
    obs, info = env.reset()
    score = 0
    done = False

    while not done:
        action = policy(obs)
        state, reward, done, truncated, info = env.step(action)
        score += reward
    else:
        print(score)
        env.reset()
