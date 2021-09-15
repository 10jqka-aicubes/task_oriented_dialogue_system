# 自适应业务逻辑更新的任务型对话算法

​	任务型对话系统(Task Oriented Dialogue System)是当下自然语言处理领域的研究热点。近些年随着技术的进步, 任务型对话系统在零售、出行、医疗、餐饮等行业的应用越来越普遍和深入。企业客服通常会涉及售前产品咨询、售后服务支持等话题。智能客服在应用中往往会面临以下挑战: 1. 业务复杂且频繁更新；2. 用户表述偏口语化不能准确描述问题；上述问题都会影响智能客服的交互体验。

- 本代码是该赛题的一个基础demo，仅供参考学习。


- ​比赛地址：http://contest.aicubes.cn/	


- 时间：2021-09 ~ 2021-10



## 如何运行Demo

- clone代码


- 准备预训练模型

  - 下载模型 [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)
  - 将所有文件放在一个目录下

- 准备环境

  - cuda10.0以上
  - python3.7以上
  - 安装python依赖

  ```
  python -m pip install -r requirements.txt
  ```

- 准备数据，从[官网](http://contest.aicubes.cn/#/detail?topicId=25)下载数据

  - 训练数据`train.zip`解压后，放在训练数据目录中，列表如下
    - new_train_dialogs
    - new_valid_dialogs
    - new_answer_list
  - 预测数据`test_A.zip`解压后，放在预测目录下，列表如下
    - test.json
    - answer_candidates.json

- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明

  - `task_oriented_dialogue_system/setting.conf`
  - 其他注意下`run.sh`里使用的参数，如指定预训练模型的路径

- 运行

  - 训练

  ```
  bash task_oriented_dialogue_system/train/run.sh
  ```

  - 预测

  ```
  bash task_oriented_dialogue_system/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash task_oriented_dialogue_system/metrics/run.sh
  ```




## 关于B榜自动化评测的代码提交说明

该赛题在B榜自动化评测分为两部分：

- 机器自动化评测
- 人机交互-人工评测

人机交互需要额外实现一个服务，参考代码会在后续放出



## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)