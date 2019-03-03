# Joint entity recognition and relation extraction as a multi-head selection problem

Implementation of the papers
[Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/abs/1804.07847) and 
[Adversarial training for multi-context joint entity and relation extraction](https://arxiv.org/abs/1808.06876).

# Requirements
* Ubuntu 16.04
* Anaconda 5.0.1
* Numpy 1.14.1
* Gensim 3.4.0
* Tensorflow 1.5.0
* PrettyTable 0.7.2

# 项目名：
## 对抗训练多头选择的实体识别和关系抽取的联合模型
### （多头选择的实体识别和关系抽取的联合模型）

---
## 主流程：
联合模型 先命名实体识别
#### *训练模块
1.数据加载  读取所有字符串;所以实体;
'token_id', 'token', "BIO", "relation", 'head'
为每个词生成 一个id 及一个50维度或者多少维度的向量  
2.双向lstm抽取特征 每个词的字母表征用双向lstm提取出来 最后一个维度肯定是hidden_size

---
## Task
Given a sequence of tokens (i.e., sentence), (i) give the entity tag of each word (e.g., NER) and (ii) the relations between the entities in the sentence. The following example indicates the accepted input format of our multi-head selection model:


```
0	Marc		B-PER		['N']					[0]		
1	Smith		I-PER 		['lives_in','works_for']		[5,11]
2 	lives		O		['N']					[2]
3	in		O		['N']					[3]
4	New		B-LOC		['N']					[4]
5	Orleans		I-LOC		['N']					[5] 
6	and		O		['N']					[6]
7	is		O		['N']					[7]
8	hired		O		['N']					[8]
9	by		O		['N']					[9]
10	the		O		['N']					[10]
11  government		B-ORG		['N']					[11]
12	.		O		['N']					[12]
```

## Configuration
The model has several parameters such as: 
* EC (entity classification) or BIO (BIO encoding scheme)
* Character embeddings
* Ner loss (softmax or CRF)

that could be specified in the configuration files (see [config](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/configs)).

## Run the model

```
./run.sh
```

## More details
Commands executed in ```./run.sh```:

1. Train on the training set and evaluate on the dev set to obtain early stopping epoch
```python3 train_es.py```
2. Train on the concatenated (train + dev) set and evaluate on the test set until either (1) the max epochs or (2) the early stopping limit (specified by train_es.py) is exceeded
```python3 train_eval.py```

## Notes

Please cite our work when using this software.

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Joint entity recognition and relation extraction as a multi-head selection problem. Expert Systems with Applications, Volume 114, Pages 34-45, 2018

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Adversarial training for multi-context joint entity and relation extraction, In the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018

吉安妮斯·贝考利斯、约翰内斯·德鲁、托马斯·德米斯特、克里斯·德维尔德。作为一个多头选择问题的联合实体识别和关系提取。专家系统与应用，第114卷，第34-45页，2018年



吉安妮斯·贝考利斯、约翰内斯·德鲁、托马斯·德米斯特、克里斯·德维尔德。《自然语言处理经验方法会议记录》（EMNLP）中多语境联合实体和关系提取的对抗性培训，2018年