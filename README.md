# Double Duelling Deep Q Network

![LunarLanderGif](LunarLander_DRL.gif)

## **Requirements**

### Quick start

```
pip install -r requirements.txt
python env.py
```

- Python 3.10+
- Numpy
- Pytorch
- Gymnasium
  > [Lunar Lander Environment Docs](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)

## **Deep Q Learning**

- **What is Q learning?**

  - A **Model-Free Reinforcement Learning** algorithm to learn the **Quality** value of taking an **Action** in a particular **State**.

    > [Learn more](https://en.wikipedia.org/wiki/Q-learning)

  - Following the **Bellman update equation**, we can train an agent to take high _**quality**_ actions that lead to states that maximize return in _**reward**_

    ![Bellman Equation image](https://i.gyazo.com/8f8d8ba0a9dfc15478a940834902327c.png)

  - We construct a **Q**uality table of **states** , **actions**, **rewards**, and iteratively update it with the equation above.

- **Applying Deep Learning**

  - Instead of storing a table of **state transitions**, use **neural networks** to approximate the **Q** function.
    > **Why?** When dealing with _extremely large_ or _continuous_ state spaces, storing the **Q**uality function in a table is no longer feasible.
  - **Replay Buffer**

    - **Represents the agents memory**
    - **Store transitions** on every step (state, action, reward, next_state, terminated)
    - **Circular insertion**
    - **Samples batches of transitions** for neural network training

  - **New Update Equation**:

    ![DQN](https://i.gyazo.com/5b4d3783e1812288b4af6822cbaa10c8.png)

  - **Psuedo Code**
    ![PseudoCode](https://i.gyazo.com/d5326365b7352555e3a1240515ac637c.png)
    > **Note:** Using **Mean Squared Error** for loss, and **Stochastic Gradient Descent** for back propogation

- **Modifictions**

  - **Double Deep Q Networks**

    > **Purpose:** Stabilize training

    - Use two Neural Networks
      - **Q** Network
      - **Q_target** Network
    - Calculate loss between them, with respect to some reward
    - Copy Q Network weights to Q_target Network every N iterations
    - **Architecture Diagram**
      ![DDQN](https://i.gyazo.com/d1900d691743d7b0fb95d4ea97799b6e.png)

    - **Updated Equation:**

      ![DDQN](https://i.gyazo.com/f96d4235aacd5486ea24b73e8f446a10.png)

  - **Double Duelling Deep Q Networks**

    > **Purpose:** Faster convergence

    - Using the same technique above, we change the Neural Network architecture to produce a **V**alue for being in a state and reward estimates for all possible **A**ctions (A.K.A. **Advantage**), then calculate **Q**uality
    - **Architecture diagram**

      ![DuellingDQN](https://i.gyazo.com/ae7dd07b5da3b785e5692aeae076b300.png)

    - **Updated Equation:**

      ![DQN](https://i.gyazo.com/c5e5307d7d35a8040cfd71cedbbc9a0a.png)

## **Results**

### Deep Q Network

![graphimg1](dqn_result.png)

### Double Deep Q Network

![graphimg2](ddqn_result.png)

### Double Duelling Deep Q Network

![graphimg3](dddqn_result.png)

**References:**

- [Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
- [van Hasselt, H., Guez, A. & Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning.](https://arxiv.org/pdf/1509.06461.pdf)
- [Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M. & de Freitas, N. (2015). Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
