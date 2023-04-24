# Lec11 深度强化学习
## 应用ε-贪心的蒙特卡洛算法
在应用ε-贪心的蒙特卡洛算法中，我们在策略π的基础上，在每次迭代中找到所经过的状态-动作对，然后更新Q值，最后再根据更新后的Q值来更新策略π。
注意：使用ε-贪心，意味着在更新迭代时，有一部分动作的概率是ε/动作总数量，选择最优策略的概率是1-ε。

## TD算法
TD算法和蒙特卡洛算法有一些不同：
* TD在有或者没有结果的情况下均可以学习(如不必等一盘象棋结束)；MC则必须等待结束得到结果。TD胜！
* TD在更新状态价值时使用的是TD 目标值，即基于即时奖励和下一状态的预估价值来替代当前状态在状态序列结束时可能得到的收获，是当前状态价值的有偏估计；而MC则使用实际的收获来更新状态价值，是某一策略下状态价值的无偏估计，MC胜！
* 虽然TD得到的价值是有偏估计，但是其方差却比MC得到的方差要低，且对初始值敏感，通常比蒙特卡罗法更加高效。TD胜！

**在强化学习的应用中，有两种时序差分算法，分别是SARSA算法和Q-learning算法，它们都可以被运用到实践中。**

## SARSA算法
- SARSA算法是一种基于值的强化学习算法，它的全称是State-Action-Reward-State-Action，即状态-动作-奖励-状态-动作。
- SARSA算法的核心思想是：在每一步更新Q值时，都**使用ε-贪心最优的动作来更新Q值**，即$Q(s,a)←Q(s,a)+α(r+γQ(s',a')-Q(s,a))$
- SARSA算法的伪代码如下：
```
Initialize Q(s,a),∀s∈S,a∈A(s),Q(terminal-state,·)=0
Repeat (for each episode):
    Initialize s
    Choose a from s using policy derived from Q (e.g., ε-greedy)
    Repeat (for each step of episode):
        Take action a, observe r, s'
        Choose a' from s' using policy derived from Q (e.g., ε-greedy)
        Q(s,a)←Q(s,a)+α(r+γQ(s',a')-Q(s,a))
        s←s'; a←a';
    until s is terminal
```
SARSA通过策略给定最优的第二步来估计当前最优的第一步，如果第二步的效果很好，那么Q值就会被更新为更大的值，反之则会被更新为更小的值

## Q-learning算法
- Q-learning算法是一种基于值的强化学习算法，它的全称是Quality-Learning，即质量学习。
- Q-learning算法的核心思想是：在每一步更新Q值时，都**使用下一状态下的最大动作来更新Q值**，即:$Q(s,a)←Q(s,a)+α(r+γ*argmax_aQ(s',a)-Q(s,a))$
- Q-learning算法的伪代码如下：
```
Initialize Q(s,a),∀s∈S,a∈A(s),Q(terminal-state,·)=0
Repeat (for each episode):
    Initialize s
    Repeat (for each step of episode):
        Choose a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe r, s'
        Q(s,a)←Q(s,a)+α(r+γ*argmax_aQ(s',a)-Q(s,a))
        s←s'
    until s is terminal
```
**Q-learning通过策略给定最优（不进行贪心了！）的第二步来估计当前最优的第一步**，如果第二步的效果很好，那么Q值就会被更新为更大的值，反之则会被更新为更小的值

## 梯度下降
### 策略梯度定理
对于片段性问题，定义策略π的性能为$J(θ)=v_{πθ}(S_0)$
在这里，θ是策略π的参数，v是状态价值函数，$S_0$是初始状态。θ的参数往往通过神经网络进行设定。
则策略梯度定理告诉我们，策略π的梯度为：
$∇_θJ(θ)=E_π[∑_a∇_θπ(a_t|s_t)Q_π(s_t,a_t)]$
其中，$Q_π(s_t,a_t)$是在状态$s_t$下采取动作$a_t$的价值，$∇_θπ(a_t|s_t)$是在状态$s_t$下采取动作$a_t$的概率的梯度。

### 蒙特卡洛策略梯度算法
将策略梯度定理中的公式转化为基于采样的、期望相等的近似公式
则状态$S_t$的价值变化率为：$G_t/π(a_t|s_t,θ)*∇_θπ(a_t|s_t,θ)$
状态$S_t$的价值变化率乘以衰减率γ，则可以生成一个序列，得到$S_0$的价值变化率：
$∇J(θ)=E_π[∑_tγ^tG_t/π(a_t|s_t,θ)*∇_θπ(a_t|s_t,θ)]$

下面给出蒙特卡洛策略梯度算法的伪代码：
```
Initialize θ
Repeat (for each episode):
    Generate an episode S_0,A_0,R_1,...,S_T-1,A_T-1,R_T following π(·|·,θ)
    For each step of episode t=0,1,...,T-1:
        G←∑_(k=t+1)^T*γ^(k-t-1)*R_k
        θ←θ+αγ^tG∇_θlnπ(A_t|S_t,θ)
```

### 蒙特卡洛策略梯度算法 with baseline
在蒙特卡洛策略梯度算法中，我们可以**引入一个baseline，即$b(S_t)$，来减少方差，使得算法更加稳定**。
在策略梯度定理中减去baseline之后，原定理变为：
$∇_θJ(θ)=E_π[∑_a∇_θπ(a_t|s_t)(Q_π(s_t,a_t)-b(s_t))]$
一般令$b(S_t)=v(S_t,w)$，其中另一个神经网络W的参数为w，v是状态价值函数，$S_t$是状态。
则蒙特卡洛策略梯度算法 with baseline的伪代码如下：
```
Initialize θ,w
Repeat (for each episode):
    Generate an episode S_0,A_0,R_1,...,S_T-1,A_T-1,R_T following π(·|·,θ)
    For each step of episode t=0,1,...,T-1:
        G←∑_(k=t+1)^T*γ^(k-t-1)*R_k
        δ←G-v(S_t,w)
        θ←θ+αγ^t*δ*∇_θlnπ(A_t|S_t,θ)
        w←w+αγ^t*δ*∇_wv(S_t,w)
```

### Actor-Critic算法
将蒙特卡洛策略梯度算法 with baseline中的蒙特卡洛算法替换为时序差分算法，即将全部return替换为单步return，可以获得单步Actor-Critic算法。
其中：
* Actor是按照策略梯度算法的策略$π(A_t|S_t,θ)$决定下一步的行动
* Critic是利用状态价值函数$v(S_t,w)$来评价策略的好坏

Actor-Critic算法的伪代码如下：
```
Initialize θ,w,I=1
Repeat (for each episode):
    Generate an episode S_0,A_0,R_1,...,S_T-1,A_T-1,R_T following π(·|·,θ)
    while S_t is not terminal:
        δ←R_t+γv(S_t+1,w)-v(S_t,w)
        θ←θ+αI*δ*∇_θlnπ(A_t|S_t,θ)
        w←w+αI*δ*∇_wv(S_t,w)
        I←I*γ
        S_t←S_t+1
```