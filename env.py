import gymnasium as gym
from DuellingDQN import *
import matplotlib.pyplot as plt

# if we can see evidence of learning on other openAI gym envs, we have implemented correctly
env = gym.make("LunarLander-v2", render_mode='human')
obs, _= env.reset()


agent = Agent(lr=0.01, gamma=0.99, eps=1, actions=4,
              input_shape=obs.shape, hidden_layer=512)
print(agent.Q.device)
epochs = 500
scores = []
episodes = []


def policy(obs):
    return agent.choose_action(obs)


rewards = []
for _ in range(epochs):
    terminated, truncated = (False, False)
    reward_total = 0
    steps = 0
    while not terminated and not truncated:

        action = policy(obs)
        observation, reward, terminated, truncated, info = env.step(action)

        if steps > 2000:
            # avoiding polluting the replay buffer
            terminated = True

        agent.store_transition(obs, action, reward,
                               observation, terminated)
        agent.learn()

        reward_total += reward
        obs = observation
        steps += 1
    else:
        rewards.append(reward_total)
        print(reward_total)
        # agent.soft_update()
        if epochs % 10 == 0:
            scores.append(np.mean(rewards[-10:]))

        env.reset()


plt.plot(scores)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.show()
