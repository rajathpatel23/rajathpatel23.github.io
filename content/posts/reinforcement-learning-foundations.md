---
title: "Reinforcement Learning Foundations: From MDPs to Deep Q-Learning"
date: 2025-08-13
draft: false
tags: ["machine-learning", "reinforcement-learning", "deep-learning", "mdp", "q-learning"]
categories: ["Technical", "Tutorial"]
showToc: true
TocOpen: true
math: true
cover:
    image: "/img/project_images/Know_rep.png"
    alt: "Reinforcement Learning Foundations"
    caption: "Understanding the building blocks of RL: from theory to implementation"
---

## Introduction

Reinforcement Learning (RL) has exploded in popularity — first in game-playing agents, and now in large language models via methods like RLHF (Reinforcement Learning from Human Feedback). These approaches don't just help models learn context better; they also improve reasoning by teaching them to "think in steps."

My fascination with RL began when the GPT-3 paper was published and ChatGPT emerged as the so-called "tool of the decade." I wanted to go beyond using these models — I wanted to understand *how* they work under the hood. That meant building RL concepts from the ground up: deriving equations, implementing toy solutions in environments like CartPole and FrozenLake, and seeing theory come alive in code.

This post is the first in a series where we'll start with **foundations**:

- What is an MDP and how does it differ from an HMM?
- How do rewards and value functions actually work?
- How can we move from *knowing* state values to *learning* optimal actions with Q-learning?
- And how do we scale that to Deep Q-Learning when the state space explodes?

By the end, you'll have a baseline RL toolkit — from mathematical definitions to runnable code — and a clear picture of how these pieces fit together when building agents.

## Why This Series?

When I first encountered MDPs, I assumed they were simply an extension of Hidden Markov Models. I quickly learned this was wrong. HMMs deal with hidden states and observations, while MDPs deal with **fully observable** states and decision-making under uncertainty. Understanding this difference changed how I approached RL problems.

This blog is written in a **learning-by-doing** style — meaning you'll see the concepts, math, and code side by side, along with real-world analogies (including fraud detection examples) and small "try-it-yourself" prompts.

## Part 1: Foundations with MDPs (and a "HMM aside")

### Markov Decision Process (MDP)

**Markov Decision Process (MDP)** models an agent interacting with an environment over discrete time steps. It's defined by the 5-tuple $(S, A, P, R, \gamma)$ where:

- $S$ = states
- $A$ = actions  
- $P(s'|s,a)$ = transition probability of reaching $s'$ after taking action $a$ in state $s$
- $R(s,a,s')$ = reward for this transition
- $\gamma \in [0,1)$ = discount factor for future rewards

### The Markov Property

> The future is conditionally independent of the past, given the present state.

Formally: 
$$P(s_{t+1} | s_t, a_t, s_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

### Why MDPs Matter
- They formalize **sequential decision making**: robotics, games, recommendation systems.
- They give a clear mathematical foundation for defining and solving RL problems via **value functions**, **policy search**, and **dynamic programming**.

### How do Markov Decision Processes differ from Hidden Markov Models

- HMMs deal with partial observability (you only see observations, not the actual underlying states).
- The context window for HMMs is technically just 1 since they're first-order Markov models, meaning they depend only on the current hidden state.

### HMM vs MDPs

| Aspect | HMM | MDP |
| --- | --- | --- |
| **State** | Hidden (unobserved) | Fully observed |
| **Observations** | Emitted from hidden states | Not applicable (agent directly observes state) |
| **Actions** | No actions; just a generative sequence model | Agent chooses actions $a$ in each state $s$ |
| **Transition Model** | $P(h_{t+1}|h_t)$ | $P(s_{t+1}|s_t,a_t)$ |
| **Emission/Reward** | Emission: $P(o_t|h_t)$ | Reward: $R(s_t,a_t,s_{t+1})$ |
| **Objective** | Compute likelihood or decode hidden path | Maximize expected cumulative reward |
| **Algorithms** | Forward/backward; Viterbi; Baum–Welch (EM) | Value iteration; policy iteration; Q-learning; policy gradients |
| **Use Cases** | Sequence labeling (speech, POS, bioinfo) | Sequential decision-making (robotics, games, control) |

## Reward Functions

### Definition

A **reward function** tells you how "good" a single transition is. Formally:

$$R(s,a,s') = \text{expected immediate reward when you take action } a \text{ in state } s \text{ and end up in } s'$$

| Symbol | Meaning |
| --- | --- |
| $s$ | Current state |
| $a$ | Action taken |
| $s'$ | Next state |
| $R(s,a,s')$ | Reward you get for that $(s \to s')$ transition |

### Intuition

- In a game, you might get +10 points for eating a pellet or –100 if you hit a ghost.
- In finance, you might receive +$5 if a trade succeeds or –$2 if it fails.
- In fraud detection, you might get +$10 if you correctly detect fraud, -$100 if you miss fraud, -$20 for a false positive, and +$5 if you correctly detect non-fraud.

## State-Value Function $V^\pi(s)$

Under a given policy $\pi$, the value of a state $s$ is the expected, discounted sum of future rewards when you start in $s$ and follow $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_{t+1} \mid S_0=s\right]$$

- $\gamma$ trades off immediate vs. long-term reward.
- High $V^\pi(s)$ means "good to be here under $\pi$"
- $r_{t+1} = R(S_t, A_t, S_{t+1})$
- $\mathbb{E}$ = average over all possible futures under $\pi$

## Action Value Function $Q^\pi(s,a)$

$Q^\pi(s, a)$ is the expected (average) discounted return if you take action $a$ in state $s$ now and then follow the policy $\pi$ afterwards.

### Why do we need the Action-Value function $Q(s,a)$?

**Short answer:**

Because if you only know how good a **state** is (that's $V(s)$), you still don't know which **action** to take *from that state* without either (a) a model of the world to look ahead, or (b) evaluating every action by trial. $Q(s,a)$ tells you the expected return **if you take action $a$ now** in state $s$, then behave well after. That lets you pick the best action *locally* without a model.

### Intuition (intersection analogy)

You're at a road intersection (state $s$). Knowing "this intersection is promising" (high $V(s)$) doesn't tell you whether to **turn left** or **right**. The thing you actually need at the decision point is: "If I **turn left** right now, how good is that?" That number is $Q(s,\text{left})$. Likewise for right.

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+1+k} \mid S_t=s, A_t=a\right]$$

### Action-Value as an Expectation (from returns to Bellman)

**TL;DR:**

We turn the scary infinite return into a **local recursion**: "**reward now + discounted value later**," averaged over what can happen next. This uses two facts:

- the **law of total expectation**
- the **Markov property**

**Step 1: Start from the definition (returns view)**
   
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+1+k} \mid S_t=s, A_t=a\right]$$

**Step 2: Peel off one step**
   
Separate the immediate reward from everything after:
   
$$Q^\pi(s,a) = \mathbb{E}\left[r_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a\right]$$
   
where $G_{t+1} = \sum_{k=0}^{\infty}\gamma^k r_{t+2+k}$ is "the return starting next step."

**Step 3: Condition on the next state $s'$ (law of total expectation)**
   
$$Q^\pi(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\left[\mathbb{E}\left[r_{t+1} + \gamma G_{t+1} \mid S_t=s,A_t=a,S_{t+1}=s'\right]\right]$$

**Step 4: Use the Markov property (make it local)**
   
Given $(s,a,s')$:
   
- Immediate reward depends only on that transition: $\mathbb{E}[r_{t+1}|s,a,s'] = R(s,a,s')$
- The future return depends only on the **next state** (and then following $\pi$): $\mathbb{E}[G_{t+1}|S_{t+1}=s'] = V^\pi(s')$
   
So the inner expectation becomes $R(s,a,s') + \gamma V^\pi(s')$, yielding:
   
$$Q^\pi(s,a) = \mathbb{E}_{s' \sim P(\cdot|s,a)}\big[R(s,a,s') + \gamma V^\pi(s')\big]$$

**Step 5: Expand $V^\pi$ over next actions**
   
By definition of the state value: $V^\pi(s') = \sum_{a'} \pi(a'|s') Q^\pi(s',a')$
   
Putting it all together gives the **Bellman expectation equation for $Q^\pi$**:
   
$$\boxed{Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')\Big]}$$

### Simple toy example

From state $s$, action $a$ leads to:
- $s_1$ with prob 0.7, reward 5
- $s_2$ with prob 0.3, reward 0

Let $\gamma=0.9$, and suppose $V^\pi(s_1)=10$ and $V^\pi(s_2)=2$. Then:

$$\begin{align}
Q^\pi(s,a) &= 0.7(5 + 0.9 \times 10) + 0.3(0 + 0.9 \times 2) \\
&= 0.7(14) + 0.3(1.8) \\
&= 9.8 + 0.54 \\
&= \mathbf{10.34}
\end{align}$$

It's a **weighted average** of "reward now + discounted value later" over possible next states.

## Q-Learning

Knowing **state value** is great—if you also have a **model** of the world to look ahead. But when you don't, you want to know directly: *How good is taking action a in state s right now?* That's the **action-value** ($Q(s,a)$), and **Q-learning** learns it **from experience** with no model required.

### Tabular Q-Learning — from optimality to an update you can code

#### 1) Objective: optimal action-values

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\Big[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\Big]$$

This is the **Bellman optimality equation**. If we had $Q^*$, the greedy policy $\pi^*(s) = \operatorname*{argmax}_a Q^*(s,a)$ is optimal.

#### 2) One-step TD target (what we aim at each step)

Replace the expectation over $s'$ with a **sample** from the environment:

$$y_{\text{target}} = r + \gamma \max_{a'} Q(s',a')$$

This is a sampled estimate of the right-hand side of Bellman optimality.

#### 3) Q-learning update (move toward the target)

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]$$

- $\alpha$ is the learning rate
- We **bootstrap** using our own $Q$ on $s'$
- Using $\max_{a'}$ (rather than the next action actually taken) makes Q-learning **off-policy**: it learns about the greedy policy even if behavior is exploratory

#### 4) Exploration: ε-greedy behavior policy

We still need to **visit** state-actions to learn them:
- With probability ε: take a random action (explore)
- Otherwise: take $\operatorname*{argmax}_a Q(s,a)$ (exploit)

Typical schedule: start ε at 1.0, decay to 0.1 (or 0.01) over many episodes.

#### 5) Tiny numeric example (feel the TD step)

Suppose at $(s,a)$ you observe:
- reward $r=1.0$, next state $s'$
- current $Q(s,a)=0.50$
- $\max_{a'} Q(s',a') = 0.80$
- $\gamma=0.9$, $\alpha=0.5$

Target: $y = r + \gamma \max_{a'} Q(s',a') = 1.0 + 0.9 \times 0.80 = 1.72$

Update: $Q_{\text{new}}(s,a) = 0.50 + 0.5 \times (1.72 - 0.50) = \mathbf{1.11}$

You've moved **halfway** toward the target.

#### 6) SARSA vs Expected SARSA vs Q-learning (quick contrast)

- **SARSA (on-policy)** uses the **actual next action** $a' \sim \pi$: 
  $$Q \leftarrow Q + \alpha[r + \gamma Q(s',a') - Q]$$
- **Expected SARSA** computes the **average** over next actions: 
  $$Q \leftarrow Q + \alpha[r + \gamma \sum_{a'} \pi(a'|s')Q(s',a') - Q]$$
- **Q-learning (off-policy)** uses the **max** over actions: 
  $$Q \leftarrow Q + \alpha[r + \gamma \max_{a'} Q(s',a') - Q]$$

#### 7) Minimal NumPy implementation (FrozenLake, tabular)

```python
import numpy as np, gym

env = gym.make('FrozenLake-v1', is_slippery=False)
nS, nA = env.observation_space.n, env.action_space.n

Q = np.zeros((nS, nA))
alpha, gamma = 0.8, 0.9

eps, eps_min, eps_decay = 1.0, 0.1, 0.995
episodes = 2000

def eps_greedy(s):
    if np.random.rand() < eps:
        return env.action_space.sample()
    return np.argmax(Q[s])

for ep in range(episodes):
    s, done = env.reset(), False
    while not done:
        a = eps_greedy(s)
        s2, r, done, _ = env.step(a)
        target = r + (0 if done else gamma * np.max(Q[s2]))
        Q[s, a] += alpha * (target - Q[s, a])
        s = s2
    eps = max(eps_min, eps * eps_decay)

print("Q-table:\n", Q)
```

> **Terminal states**: if done, set the bootstrapped term to 0 (as above).

## Deep Q-Learning

Q-learning works only to an extent where it is possible to keep track of states and actions at a given time. However, as we move away from simple games like tic-tac-toe to something as complex as Chess, the number of states and actions at a given time increases exponentially. At that stage, keeping track of a Q-table becomes very compute intensive, doesn't scale well, and is very slow. Hence, the paradigm of Deep Q-learning provides the potential to overcome this barrier.

Tabular methods break when the state space is huge. DQN replaces the table with a neural net $Q_\theta(s,a)$ that outputs all action values from a state.

### Core ideas (why DQN works)

1. **Function approximation**: $Q_\theta(s,a)$ via a neural net
2. **Experience replay**: Store $(s,a,r,s',\text{done})$ transitions, train on random mini‑batches to break correlation

### Loss and targets

For a batch $\mathcal{B}$ of transitions:

$$y_i = \left\{
\begin{array}{ll}
r_i & \text{if } \text{done}_i \\
r_i + \gamma \max_{a'} Q_{\theta^-}(s'_i, a') & \text{otherwise}
\end{array}
\right.$$

Squared loss (often Huber in practice):

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \big( y_i - Q_\theta(s_i, a_i) \big)^2$$

Gradient step: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$

**Target network update** (periodic hard update every $C$ steps): $\theta^- \leftarrow \theta$

(or soft update: $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$)

### Double DQN (reduces overestimation bias)

Use the **online** net to pick the action and the **target** net to evaluate it:

$$y_i = r_i + \gamma Q_{\theta^-}\left(s'_i, \operatorname*{argmax}_{a'} Q_\theta(s'_i,a')\right)$$

### Code Example for Deep Q-Learning

For a complete implementation example, check out this [GitHub gist](https://gist.github.com/rajathpatel23) with a working DQN on CartPole.

---

## What's Next?

In the next post, we'll dive deeper into:
- Policy Gradient methods (REINFORCE, Actor-Critic)
- Advanced DQN variants (Dueling DQN, Prioritized Experience Replay)
- Real-world applications in fraud detection and recommendation systems

Stay tuned for more hands-on RL content! 🚀

---

*Have questions or thoughts about this post? Feel free to [reach out](mailto:rpatel12@umbc.edu) - I'd love to discuss RL concepts and applications!* 