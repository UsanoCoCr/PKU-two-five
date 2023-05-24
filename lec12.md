# Lec12

对于transformer，可以参考以下文章：
* [The Illustrated Transformer](https://zhuanlan.zhihu.com/p/338817680)
  
  
## 自注意力机制
在词嵌入的过程中，为了确定在某个词上所放的注意力，可以引入自注意力机制：
* 查询矩阵Q
* 键矩阵K
* 值矩阵V

```c++
attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k))V
```
其中，d_k是K的维度。

![self-attention](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9nNG5lcWpVQ2xBNlB6dFNDdTlnTUkxZjBrdmFsanZ1bTVXUE9Fd1JKUmRERnV0aWJOaWJkWVVIRVFxVHNKWmlhM2piYTlKMkpZc3lYS2ZqSEVockVKOXhIUS82NDA?x-oss-process=image/format,png)

缩放：将K和Q的点积结果除以sqrt(d_k)，这样做的目的是为了避免点积结果过大，导致softmax函数的梯度过小。

### 打分函数
加性注意力是一种用于计算和查询矩阵K和值矩阵V之间的注意力的打分函数

## 多头注意力
多头注意力是h个self-attention的组合。映射到h个空间，关注不同的特征维度。
最后将h个注意力汇聚的输出拼接在一起，并且通过另一个可以学习的线性投影进行变换，产生最终输出。

## Transformer
Transformer是一个基于注意力机制的序列到序列模型，用于解决机器翻译问题。
Transformer的整体结构由编码器和解码器组成，编码器和解码器都是由相同的层组成，每层由多头注意力机制和全连接前馈神经网络组成，层之间通过残差连接和层归一化连接。
![transformer](https://pic4.zhimg.com/v2-f8d834e3e3387b67f81c8c8c20c3c34d_1440w.jpg?source=172ae18b)

在transformer中，也可以将多层encoder和多层decoder串联起来，这样就可以得到更深的网络结构。
在解码器中，encoder的输出没有直接作为decoder的输入，解码器的输入被称为目标序列，输入一个特殊的token（目标序列开始的token），输出一个单词，然后将这个单词作为下一个时刻的输入，直到输出一个特殊的token（目标序列结束的token）。

## NLP的预训练模型
预训练模型用于在无标签数据上学习通用语言表示，从而提高NLP任务的性能
常见的预训练模型有：
* GPT：基于transformer的语言模型，学习单向上下文信息
* BERT：基于transformer的双向语言模型，学习双向上下文信息

参数规模提升带来能力“涌现Emergent”，模型的进步是阶梯型的
