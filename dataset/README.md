This dataset is developed within the following paper:

*Weiwei Sun, Shuo Zhang, Krisztian Balog, Zhaochun Ren, Pengjie Ren, Zhumin Chen, Maarten de Rijke. "Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems". In SIGIR.* [Paper link](https://arxiv.org/pdf/2105.03748)


**Note: In our work (Multi-task User Simulator), we do not use ReDial and JDDC (chinese language dataset).**


## Data

The dataset (see [dataset](https://github.com/sunnweiwei/user-satisfaction-simulation/tree/master/dataset)) is provided a TXT format, where each line is separated by "\t": 

- speaker role (USER or SYSTEM), 
- text, 
- action, 
- satisfaction (repeated annotation are separated by ","), 
- explanation text (only for JDDC at dialogue level, and repeated annotation are separated by ";")

And sessions are separated by blank lines.

Since the original dataset does not provide actions, we use the action annotation provided by [IARD](https://github.com/wanlingcai1997/umap_2020_IARD) and included it in *ReDial-action.txt*.

The JDDC data set provides the action of each user utterances, including 234 categories. We compress them into 12 categories based on a manually defined classification method (see *JDDC-ActionList.txt*).

## Data Statistics

The USS dataset is based on five benchmark task-oriented dialogue datasets: [JDDC](https://arxiv.org/abs/1911.09969), [Schema Guided Dialogue (SGD)](https://arxiv.org/abs/1909.05855), [MultiWOZ 2.1](https://arxiv.org/abs/1907.01669), [Recommendation Dialogues (ReDial)](https://arxiv.org/abs/1812.07617), and [Coached Conversational Preference Elicitation (CCPE)](https://www.aclweb.org/anthology/W19-5941.pdf).

| Domain      |    JDDC |     SGD | MultiWOZ |  ReDial |    CCPE |
| ----------- | ------: | ------: | -------: | ------: | ------: |
| Language    | Chinese | English |  English | English | English |
| #Dialogues  |   3,300 |   1,000 |    1,000 |   1,000 |     500 |
| Avg# Turns  |    32.3 |    26.7 |     23.1 |    22.5 |    24.9 |
| #Utterances |  54,517 |  13,833 |   12,553 |  11,806 |   6,860 |
| Rating 1    |     120 |       5 |       12 |      20 |      10 |
| Rating 2    |   4,820 |     769 |      725 |     720 |   1,472 |
| Rating 3    |  45,005 |  11,515 |   11,141 |   9,623 |   5,315 |
| Rating 4    |   4,151 |   1,494 |      669 |   1,490 |      59 |
| Rating 5    |     421 |      50 |        6 |      34 |       4 |

## Baselines

![Performance for user satisfaction prediction. Bold face indicates the best result in terms of the corresponding metric. Underline indicates comparable results to the best one.](https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/imgs/satisfaction-prediction.png)

![ Performance for user action prediction. Bold face indicates the best result in terms of the corresponding metric. Underline indicates comparable results to the best one.](https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/imgs/action-prediction.png)

## Cite

```
@inproceedings{Sun:2021:SUS,
  author =    {Sun, Weiwei and Zhang, Shuo and Balog, Krisztian and Ren, Zhaochun and Ren, Pengjie and Chen, Zhumin and de Rijke, Maarten},
  title =     {Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems},
  booktitle = {Proceedings of the 44rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series =    {SIGIR '21},
  year =      {2021},
  publisher = {ACM}
}
```
