# ChatGPT提示工程

## 1.课程介绍
大语言模型LLM一般可以分为两种：
* 基础LLM：对于训练数据，通过前面的单词预测后面的单词。
* 指令学习LLM：能够遵循指令，而非预测后面的单词。   
    * 从基础LLM上再一步训练
    * 使用RLHF（Reinforcement Learning with Human Feedback）的技术进一步训练。 

## 2.提示性语言
在使用提示性语言为GPT输入时，需要遵守两个原则：
* 写出清晰而具体的提示
* 给模型思考的时间

可以使用openai的OpenAI库来引用chatGPT的API.

### 写出具体而清晰的提示
* 使用分隔符明确指出输入的不同部分
* 要求结构化的输出，如Html或Json
* 要求模型检查任务条件是否满足（若文档和客户要求相互矛盾，添加异常处理）
* 添加几句提示语（模型在执行之前给他提供几个成功的例子）

### 给模型思考的时间
* 指定完成一项任务所需的步骤
* 让模型在得出结论之前给出自己的结果（例：学生回答数学问题，自己先做，再对比自己和学生的答案）

### 模型的局限性
* 模型可能会回答一些虚无缥缈的问题，导致产生“幻觉”（回答一款不存在的产品）
    * 减少幻觉：基于文本、相关信息再来回答

## 3.工程迭代
提示词的使用：
* 简单、清晰
* 分析为什么结果和预期结果有出入
* 重新定义提示词
* 重复上述步骤

## 4.文本总结

