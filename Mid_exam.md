Author:UsanoCoCr
Date:2023.4.19
仅为北京大学人工智能基础复习使用
# 目录
- [目录](#目录)
- [全局搜索](#全局搜索)
  - [全局搜索模型的描述](#全局搜索模型的描述)
  - [图搜索与树搜索](#图搜索与树搜索)
  - [搜索算法的评价标准](#搜索算法的评价标准)
  - [无信息搜索](#无信息搜索)
    - [宽度优先搜索bfs](#宽度优先搜索bfs)
    - [深度优先搜索dfs](#深度优先搜索dfs)
      - [深度受限的深度优先搜索](#深度受限的深度优先搜索)
    - [双向搜索](#双向搜索)
    - [一致代价搜索](#一致代价搜索)
  - [有信息搜索](#有信息搜索)
    - [贪婪最佳搜索](#贪婪最佳搜索)
    - [A\*搜索](#a搜索)
    - [常见的启发式函数](#常见的启发式函数)
- [局部搜索](#局部搜索)
  - [爬山算法](#爬山算法)
    - [随机爬山法](#随机爬山法)
    - [第一选择爬山法](#第一选择爬山法)
    - [随机重启爬山法](#随机重启爬山法)
  - [模拟退火算法](#模拟退火算法)
  - [局部束搜索](#局部束搜索)
  - [遗传算法](#遗传算法)
  - [梯度下降法](#梯度下降法)
- [对抗搜索](#对抗搜索)
  - [极大极小值搜索 Minimax](#极大极小值搜索-minimax)
  - [Alpha-Beta剪枝](#alpha-beta剪枝)
  - [不完美的实时决策](#不完美的实时决策)
  - [蒙特卡洛树搜索 MCTS](#蒙特卡洛树搜索-mcts)
    - [选择](#选择)
    - [扩张](#扩张)
    - [模拟](#模拟)
    - [反向传播](#反向传播)
    - [总结](#总结)
- [强化学习](#强化学习)
  - [强化学习的环境](#强化学习的环境)
    - [累积收益](#累积收益)
    - [状态价值](#状态价值)
    - [动作价值](#动作价值)
  - [寻找最优策略的方法](#寻找最优策略的方法)
    - [贪心策略](#贪心策略)
    - [ε-贪心策略](#ε-贪心策略)
    - [乐观初值贪心](#乐观初值贪心)
    - [UCB](#ucb)
  - [强化学习的算法](#强化学习的算法)
    - [贝尔曼方程](#贝尔曼方程)
    - [策略迭代](#策略迭代)
- [深度学习基础](#深度学习基础)
- [卷积神经网络 CNN](#卷积神经网络-cnn)
- [生成对抗网络 GAN](#生成对抗网络-gan)
- [循环神经网络 RNN](#循环神经网络-rnn)


# 全局搜索
## 全局搜索模型的描述
全局搜索可以用来解决一定的问题，描述一个问题模型一般有以下几个要素：
1.初始状态state0
2.可选择动作action_set
3.状态转移函数next_state=state0+action
4.目标状态state_goal
5.花费函数cost=cost(state,action)

## 图搜索与树搜索
一般情况下，搜索分为树搜索和图搜索两种，树搜索是指搜索的过程中不会出现环，图搜索是指搜索的过程中可能出现环，这两种搜索的区别在于**图搜索需要记录已经搜索过的节点**，以避免重复搜索，而树搜索不需要记录已经搜索过的节点，因为树搜索不会出现环，所以不需要记录已经搜索过的节点。

下面给出树搜索和图搜索的伪代码：
<details>
<summary>树搜索</summary>

```c++
void tree_search(state0,action_set,next_state,cost,state_goal)
{
    if(state0==state_goal)//判断是否到达目标状态
        return state0;
    for(action in action_set)//遍历所有可选择的动作
    {
        next_state=state0+action;//计算下一个状态
        tree_search(next_state,action_set,next_state,cost,state_goal);//递归搜索
    }
}
```
</details>

<details>
<summary>图搜索</summary>

```c++
void graph_search(state0,action_set,next_state,cost,state_goal)
{
    explored=empty;//初始化explored
    if(state0==state_goal)//判断是否到达目标状态
        return state0;
    for(action in action_set)//遍历所有可选择的动作
    {
        next_state=state0+action;//计算下一个状态
        if(next_state not in explored)//判断下一个状态是否已经搜索过
        {
            graph_search(next_state,action_set,next_state,cost,state_goal);//递归搜索
            explored.add(next_state);//将下一个状态加入已搜索状态集合
        }
    }
}
```
</details>

*上述树搜索/图搜索有可能陷入死循环，在这种情况下我们需要加入判定条件，当搜索的次数超过一定次数时，就停止搜索，这样就可以避免陷入死循环。*

## 搜索算法的评价标准
搜索算法的评价标准一般有以下几个：
1.搜索的时间复杂度
2.搜索的空间复杂度
3.搜索的完备性：当解存在的时候，算法是否一定能够找到解？
4.搜索的最优性：当解存在的时候，算法是否一定能够找到最优解？

**在搜索问题中，时间复杂度经常用搜索树展开的节点数目表示
空间复杂度经常用同一时间需要储存的最大节点数目来估计**

## 无信息搜索
### 宽度优先搜索bfs
宽度优先搜索的时间复杂度为O($b^d$),空间复杂度为O($b^d$)
其中b为分支因子（一个节点有多少个子节点），d为搜索深度
在使用c++实现bfs时，一般会使用queue结构，闭节点集合一般使用set结构，这样可以避免重复搜索。

<details>
<summary>宽度优先搜索bfs</summary>

```c++
void bfs(state0,action_set,next_state,cost,state_goal)
{
    queue=empty;//初始化queue
    explored=empty;//初始化explored
    queue.push(state0);//将初始状态加入queue
    while(queue not empty)//当queue不为空时
    {
        state=queue.pop();//从queue中取出一个状态
        if(state==state_goal)//判断是否到达目标状态
            return state;
        for(action in action_set)//遍历所有可选择的动作
        {
            next_state=state+action;//计算下一个状态
            if(next_state not in explored)//判断下一个状态是否已经搜索过
            {
                queue.push(next_state);//将下一个状态加入queue
                explored.add(next_state);//将下一个状态加入已搜索状态集合
            }
        }
    }
}
```
</details>

### 深度优先搜索dfs
深度优先搜索的时间复杂度为O($b^m$),空间复杂度为O($bm$)
其中b为分支因子（一个节点有多少个子节点），m为搜索树的最大深度
在使用c++实现dfs时，一般会使用stack结构
深度优先搜索在内存受限的情况下可以搜的更深，所以**在内存受限的情况下，深度优先搜索比宽度优先搜索更好**。

<details>
<summary>深度优先搜索dfs</summary>
    
```c++
void dfs(state0,action_set,next_state,cost,state_goal)
{
    stack=empty;//初始化stack
    explored=empty;//初始化explored
    stack.push(state0);//将初始状态加入stack
    while(stack not empty)//当stack不为空时
    {
        state=stack.pop();//从stack中取出一个状态
        if(state==state_goal)//判断是否到达目标状态
            return state;
        for(action in action_set)//遍历所有可选择的动作
        {
            next_state=state+action;//计算下一个状态
            if(next_state not in explored)//判断下一个状态是否已经搜索过
            {
                stack.push(next_state);//将下一个状态加入stack
                explored.add(next_state);//将下一个状态加入已搜索状态集合
            }
        }
    }
}
```
</details>

#### 深度受限的深度优先搜索
dfs因为没有存储过搜索过的点，所以可能会陷入死循环
深度受限的深度优先搜索的时间复杂度为O($b^l$),空间复杂度为O($bl$)
其中b为分支因子（一个节点有多少个子节点），l为限制层数

<details>
<summary>深度受限的深度优先搜索</summary>
    
```c++
void graph_search(state0,action_set,next_state,cost,state_goal,limit)
{
    if(state0==state_goal)
        return state0;
    if(limit==0)//判断是否到达限制层数
        return cutoff;
    cutoff_occurred=false;//判断是否陷入死循环
    for(action in action_set)
    {
        next_state=state0+action;
        result=graph_search(next_state,action_set,next_state,cost,state_goal,limit-1);//递归搜索
        if(result==cutoff)
            cutoff_occurred=true;
        else if(result!=failure)
            return result;
    }
    if(cutoff_occurred)
        return cutoff;
    else//程序陷入死循环
        return failure;
}
```
</details>
深度受限搜索引出了迭代加深算法：在深度受限的深度优先搜索中，我们可以将限制层数从0开始逐渐增加，直到找到解为止。

### 双向搜索
双向搜索的时间复杂度为O($b^{d/2}$),空间复杂度为O($b^{d/2}$)

### 一致代价搜索
一致代价搜索在c++实现中使用了优先队列的结构，通过节点到起点的cost来排列，当终点从优先队列中输出时，即找到最优解

下面是一致代价搜索的伪代码：
<details>
<summary>一致代价搜索</summary>
    
```c++
void uniform_cost_search(state0,action_set,next_state,cost,state_goal)
{
    priority_queue=empty;//初始化优先队列
    explored=empty;//初始化explored
    priority_queue.push(state0);//将初始状态加入优先队列
    while(priority_queue not empty)//当优先队列不为空时
    {
        state=priority_queue.pop();//从优先队列中取出一个状态
        if(state==state_goal)//判断是否到达目标状态
            return state;
        for(action in action_set)//遍历所有可选择的动作
        {
            next_state=state+action;//计算下一个状态
            if(next_state not in explored)//判断下一个状态是否已经搜索过
            {
                priority_queue.push(next_state);//将下一个状态加入优先队列
                explored.add(next_state);//将下一个状态加入已搜索状态集合
            }
        }
    }
}
```
</details>

## 有信息搜索
有信息搜索在搜索时会使用启发式函数f
**可采纳性：f不会高估到达目标的代价，即f(n)<=c(n,n_goal)
一致性：对启发式函数，如果从状态n1到n2的代价为c，那么f(n1)<=c+f(n2)**

### 贪婪最佳搜索
有信息搜索在搜索时会使用启发式函数h(n)，同样使用了优先队列的结构，通过节点到终点的启发式函数值来排列

### A*搜索
A*搜索在c++实现中使用了优先队列的结构，通过节点到起点的cost加上节点到终点的启发式函数值来排列
即：f(n)=g(n)+h(n)
A*搜索的实际效果取决于启发式函数的好坏
**如果启发式函数f(n)是可采纳的，那么A*搜索是最优的（即一定可以找到最优解）**

**估值函数是乐观的，A*就是最优的**

### 常见的启发式函数
* 曼哈顿距离
* 欧拉距离
* 对角线距离
* 最大距离
* 最小距离
启发式函数不必完全接近真实值，有时我们可以松弛启发性函数的值


# 局部搜索

## 爬山算法
爬山算法每次会从当前状态移动到相邻节点中最好的一个，算法会在山峰/山谷处停下，算法并不储存一棵搜索树，只存储当前节点和估值函数

下面给出爬山算法的伪代码：
<details>
<summary>爬山算法</summary>
    
```c++
void hill_climbing(state0,action_set,next_state,cost,state_goal)
{
    state=state0;//初始化当前状态
    while(true)//循环
    {
        next_state=best_next_state(state,action_set);//找到最好的邻居节点
        if(next_state.value<=state.value)//如果所有邻居节点的值都比当前节点的值小，说明已经到达山顶
            return state;
        state=next_state;//更新当前状态
    }
}
```
</details>

爬山算法可能存在的问题：
* 陷入局部最优：到达局部极值点就僵住不动了，可能达不到全局最优解
解决方法：
1.随机平移：在局部最优点附近随机选择一个点，然后继续搜索
2.随机重启：在整个搜索空间中随机选择一个点，然后继续搜索
针对上述可能存在的问题，爬山法衍生出了很多变种

### 随机爬山法
随机爬山法在爬山算法的基础上，每次从当前状态的所有更优的邻居节点中随机选择一个节点作为下一个状态，随机爬山法收敛的更慢，但是往往可以找到更优的解

### 第一选择爬山法
第一选择爬山法每次随机找一个邻居，如果邻居估值更优，则直接跳到邻居，否则继续找下一个邻居，直到找到一个更优的邻居，或者所有邻居都不比当前状态更优
此方法**不需要计算所有邻居节点的估值，在一个状态有特别多个邻居节点的时候，可以大大减少计算量**

### 随机重启爬山法
上述爬山法变种虽然可以提高爬山法的效率，但是也不具有完备性，有可能会陷在局部极值
随机重启爬山法保证了爬山过程中，如果找不到解，则从一个随机位置再次开始爬山，直到找到解为止

下面给出随机重启爬山法的伪代码：
<details>
<summary>随机重启爬山法</summary>

```c++
void hill_climbing(state0,action_set,next_state,cost,state_goal)
{
    state=state0;//初始化当前状态
    while(true)//循环
    {
        next_state=best_next_state(state,action_set);//找到最好的邻居节点
        if(next_state.value<=state.value)//如果所有邻居节点的值都比当前节点的值小，说明已经到达山顶
            return state;
        state=next_state;//更新当前状态
    }
}

void random_restart_hill_climbing(state0,action_set,next_state,cost,state_goal)
{
    while(true)//循环
    {
        state=hill_climbing(state0,action_set,next_state,cost,state_goal);//调用爬山算法
        if(state==state_goal)//判断是否到达目标状态
            return state;
        state0=random_state();//从随机位置开始爬山
    }
}
//在使用随机重启爬山法时，可以设置一个最大迭代次数，如果超过最大迭代次数还没有找到解，则返回当前找到的最优解
```
</details>

## 模拟退火算法
模拟退火算法设置了一个概率函数p，用来决定是否接受一个更差的解，**当前节点有概率p(t0)跳到一个比他更差的邻居节点。p的值随着迭代次数的增加而减小**，模拟退火算法的伪代码如下：
<details>
<summary>模拟退火算法</summary>

```c++
void simulated_annealing(state0,action_set,next_state,cost,state_goal)
{
    state=state0;//初始化当前状态
    t0=100;//初始温度
    while(true)//循环
    {
        next_state=random_next_state(state,action_set);//随机选择一个邻居节点
        if(next_state.value<=state.value)//邻居节点更差
        {
            if(p(t0)>rand())//如果概率大于随机数，则接受更差的解
                state=next_state;
        }
        else//邻居节点更优
            state=next_state;
        t0=t0-1;//温度减小
    }
}
```
</details>

## 局部束搜索
局部束搜索在同一时刻保留了k个状态，每个状态生成b个后继，如果k*b个后继中存在全局最优解，则直接返回，否则，从k*b个后继中选择k个最优的后继，作为下一时刻的k个状态，继续搜索

局部束搜索和随机重启爬山法的区别：
* 随机重启爬山法每次搜索是独立的，而**局部束搜索将k个后继节点作为一个整体，每次搜索在这k个后继节点中进行挑选**

## 遗传算法
遗传算法模拟了生物有性生殖、变异、自然选择的过程，通过不断的迭代，最终得到一个最优解
在遗传算法中会使用到函数：
1.random_state()：随机生成一个状态
2.reproduce(state1,state2)：将两个状态交配
3.mutate(state)：对状态进行变异
4.evaluate(state)：评估状态的优劣

遗传算法的伪代码如下：
<details>
<summary>遗传算法</summary>

```c++
auto reproduce(state1,state2)
{
    new_state=empty;
    c=rand();//随机选择一个数
    new_state.add(substring(state1,0,c));//将state1的前c个元素加入新状态
    new_state.add(substring(state2,c+1,size-1));//将state2的后size()-c个元素加入新状态
    return new_state;
}

void genetic_algorithm(state0,action_set,next_state,cost,state_goal)
{
    population_size=100;//种群大小
    population=init_population(population_size);//初始化种群
    while(true)//循环
    {
        new_population=empty;
        for(i=0;i<population_size;i++)
        {
            x=random_state();//随机选择一个状态
            y=random_state();//随机选择一个状态
            z=reproduce(x,y);//交配
            if (rand()<0.1)//有10%的概率进行变异
                z=mutate(z);
            new_population.add(z);//将新生成的状态加入种群
        }
        population=new_population;//更新种群
    }
    best_state=best_state(population);//找到最优解
}
```
</details>

## 梯度下降法
梯度下降法在实现上和爬山法非常相似，梯度下降即连续集里的爬山算法

# 对抗搜索
ai基础课所讲的对抗搜索主要运用在双人零和完全信息游戏中，在这一类游戏中，我们可以通过搜索来描述游戏模型：
* 初始状态：游戏开始时的状态
* Players：轮到哪个玩家进行操作
* Actions：每个玩家可以进行的操作
* 状态转移模型：每个玩家进行操作后，游戏的状态如何转移
* 终止条件：游戏何时结束
* 效用函数：评估游戏结束后两位玩家的得分

## 极大极小值搜索 Minimax
Minimax算法中双方作为理性决策者，都会选择让自己收益最大的节点
Minimax算法的时间复杂度是O(b^m)，空间复杂度是O(bm)，
其中b为每个节点的分支数，m为搜索深度

Minimax算法的伪代码如下：
<details>
<summary>Minimax算法</summary>

```c++
auto minimax(state,player)
{
    if(state is terminal)//判断是否为终止状态
        return utility(state);//返回效用值
    if(player==MAX)//如果是MAX玩家
    {
        v=-inf;//v为最大值
        for each action in actions(state)//遍历所有的操作
        {
            v=max(v,minimax(result(state,action),MIN));//更新v
        }
        return v;
    }
    else//如果是MIN玩家
    {
        v=inf;//v为最小值
        for each action in actions(state)//遍历所有的操作
        {
            v=min(v,minimax(result(state,action),MAX));//更新v
        }
        return v;
    }
}
```
</details>

## Alpha-Beta剪枝
Alpha-Beta剪枝是对Minimax算法的优化，它可以减少搜索的节点数，从而减少搜索的时间
Alpha-Beta剪枝剪去的节点一般拥有这样的特点：
* 当该节点为最大节点时，节点上方的路径要取最小值，下方的路径要取最大值，当节点的子节点传上来一个value=v，且v大于节点的兄弟节点的value=β时，我们就不用展开这个节点了，因为节点上方的取最小值路径一定会选择兄弟节点的路径
* 当该节点为最小节点时，节点上方的路径要取最大值，下方的路径要取最小值，当节点的子节点传上来一个value=v，且v小于节点的兄弟节点的value=α时，我们就不用展开这个节点了，因为节点上方取最大值的路径一定会选择兄弟节点的路径

Alpha-Beta剪枝算法的伪代码如下：
<details>
<summary>Alpha-Beta剪枝算法</summary>

```c++
auto alphabeta(state,player,alpha,beta)
{
    if(state is terminal)//判断是否为终止状态
        return utility(state);//返回效用值

    //注意：这里的最大值最小值节点是三层中的最上层
    function max_value(state,alpha,beta)
    {
        v=-inf;//v为最大值
        for each action in actions(state)//遍历所有的操作
        {
            v=max(v,min_value(result(child_state,action),alpha,beta));//更新v
            if(v>=beta)//如果v大于beta
                return v;//返回v
            alpha=max(alpha,v);//更新alpha
        }
        return v;
    }

    function min_value(state,alpha,beta)
    {
        v=inf;//v为最小值
        for each action in actions(state)//遍历所有的操作
        {
            v=min(v,max_value(result(child_state,action),alpha,beta));//更新v
            if(v<=alpha)//如果v小于alpha
                return v;//返回v
            beta=min(beta,v);//更新beta
        }
        return v;
    }
}
```
</details>

## 不完美的实时决策
不完美的实时决策是指在决策过程中，我们并不知道游戏的终止条件，因此我们需要**在决策过程中不断的评估当前的状态，使用启发式函数来辅助搜索**
当到了一定的时间/深度，则停止搜索，返回当前的最优解，这种方法是一种蒙特卡洛树搜索的近似方法

## 蒙特卡洛树搜索 MCTS
蒙特卡洛搜索是在一定的预设条件（时间、内存、迭代次数）下不断地迭代进行更新，并利用统计学原理找到近似最优解的一种方法
**蒙特卡洛的每次迭代包括四个步骤：选择、扩张、模拟、反向传播**

### 选择
选择是指从根节点开始，不断地选择下一个节点，直到遇到一个未被访问的节点，或者遇到一个终止节点
选择部分的代码是：
<details>
<summary>选择</summary>

```c++
function select(node)
{
    while(node is not terminal)//如果节点不是终止节点
    {
        if(node is fully expanded)//如果节点已经被扩张
        {
            node=best_child(node,ucb1) //选择最优子节点
            return node
        }
        else//如果节点未被扩张
        {
            expand(node)//扩张节点
        }
    }
}
```
</details>

### 扩张
扩张是指从当前节点中选择未被访问的子节点，将其加入到当前节点的子节点中
扩张部分的代码是：
<details>
<summary>扩张</summary>

```c++
function expand(node)
{
    if(node is not fully expanded)//如果节点未被扩张
    {
        action=untried_actions(node)//获取未被访问的子节点
        child_node=child_node(node,action)//获取子节点
        add_child(node,child_node)//将子节点加入到当前节点的子节点中
    }
}
```
</details>

### 模拟
模拟是指从当前节点开始，不断地随机选择一个子节点，直到遇到一个终止节点，然后返回该终止节点的效用值
有时我们可以选择随机模拟直到终局来获得效用值，也可以通过启发式函数来获得效用值
在评估上，人们人造了一个函数来反映节点的好坏：
**$UCB = Vi + C*sqrt(ln(n)/N_i)$**

### 反向传播
反向传播是指从当前节点开始，不断地更新父节点的效用值，直到遇到根节点
反向传播部分的代码是：
<details>
<summary>反向传播</summary>

```c++
function backpropagate(node,value)
{
    while(node is not root)//如果节点不是根节点
    {
        node.visits++//节点的访问次数加一
        node.value+=value//节点的效用值加上value
        node=node.parent//节点指向父节点
    }
    //其中，value = UCB(node)
}
```
</details>

### 总结
此时，我们可以写出蒙特卡洛搜索的完整版伪代码：
<details>
<summary>蒙特卡洛搜索</summary>

```c++
function monte_carlo_tree_search(state)
{
    root=initialise(state)//初始化根节点
    while(时间未到)//如果时间未到
    {
        node=select(root)//选择节点
        value=simulate(node)//模拟节点
        backpropagate(node,value)//反向传播
    }
    return best_child(root,ucb1)//返回最优子节点
}
```
</details>

# 强化学习
强化学习在游戏中也有很广泛的应用，我们可以构建问题模型：
* 初始状态S0
* 当前玩家C
* 动作A
* 状态转移函数P
* 终止状态ST
* 奖励函数R

**注意：状态转移和奖励都不一定是确定性的，可以是一个概率分布**
在强化学习中，我们要寻找一个最优策略π，使得从初始状态S0开始，不断地执行策略π，最终达到终止状态ST，期间的奖励函数R最大化

## 强化学习的环境
### 累积收益
在计算强化学习某一路径的总价值时，我们会使用累积收益G
G的计算公式是：
**$G_t = R_{t+1} + γ*R_{t+2} + ... + γ^{T-t+1}*R_T$**
其中，$γ$是折扣因子，$γ$越大，表示对未来的奖励越看重，$γ$越小，表示对未来的奖励越不看重
### 状态价值
状态价值是指在某一状态s下，执行某一策略π，从该状态开始，不断地执行策略π，最终达到终止状态ST，期间的奖励函数R的期望值
状态价值的计算公式是：
**$V_{π}(s) = E_{π}[G_t|S_t=s]$ = $E_{π}[R_{t+1} + γ*R_{t+2} + ... + γ^{T-t+1}*R_T|S_t=s]$**
### 动作价值
动作价值是指在某一状态s下，执行某一策略π，从该状态开始，执行某一动作a，不断地执行策略π，最终达到终止状态ST，期间的奖励函数R的期望值
动作价值的计算公式是：
**$Q_{π}(s,a) = E_{π}[G_t|S_t=s,A_t=a]$ = $E_{π}[R_{t+1} + γ*R_{t+2} + ... + γ^{T-t+1}*R_T|S_t=s,A_t=a]$**

状态价值和动作价值可以通过数学的方式建立联系：
**$V_{π}(s) = Σ_a π(a|s)*Q_{π}(s,a)$**
同理，有：
**$Q_{π}(s,a) = Σ_{s'} Σ_r p(s',r|s,a)*(r + γ*V_{π}(s'))$**
**上述式子即为贝尔曼方程，它的意思是：在状态s下，执行动作a，得到状态s'和奖励r，那么在状态s下，执行动作a的价值就是 奖励r + 折扣因子*状态s'产生的价值依概率的期望值**

## 寻找最优策略的方法
### 贪心策略
贪心策略是指在每一步都选择当前状态下价值最大的动作，即：
**$a = argmax_a Q_{π}(s,a)$**

### ε-贪心策略
ε-贪心策略是指在每一步都有一定的概率选择随机动作，即：
**$a = argmax_a Q_{π}(s,a)$ with probability 1-ε**
**$a = random$ with probability ε**

### 乐观初值贪心
乐观初值贪心将所有节点动作价值的初始值设的很大，这样可以使得在初始阶段，鼓励没有被访问到的节点被访问，从而使得更多节点的价值有机会被更新，从而使得最终的策略更加优秀

### UCB
估值大的节点，我们应该去访问；访问次数少的节点，我们应该去访问
将当前的估值和访问次数结合起来，得到UCB值，UCB值越大，表示该节点的价值越大，越有可能是最优节点
UCB公式是：
**$UCB = Q_{π}(s,a) + c*sqrt(ln(N)/n)$**

## 强化学习的算法
### 贝尔曼方程
贝尔曼方程分为贝尔曼期望方程和贝尔曼最优方程
贝尔曼期望方程是指：
**$Q_{π}(s,a) = Σ_{s'} Σ_r p(s',r|s,a)*(r + γ*V_{π}(s'))$**
这个式子在介绍动作价值和状态价值的关联性时已经介绍过，下面将介绍贝尔曼最优方程：
**$Q_{π}(s,a) = Σ_{s'} Σ_r p(s',r|s,a)*(r + γ*max_{a'}Q_{π}(s',a'))$**

在强化学习中，**贝尔曼期望方程对应着策略迭代，贝尔曼最优方程对应着值迭代**

### 策略迭代


# 深度学习基础
# 卷积神经网络 CNN
# 生成对抗网络 GAN
# 循环神经网络 RNN