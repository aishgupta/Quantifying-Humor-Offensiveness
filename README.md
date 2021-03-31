# Humor@IITK at SemEval-2021 Task 7: Large Language Models for Quantifying Humor and Offensiveness

In the paper, we have proposed two different approaches;
- Single-task model
- Multi-task model

Run following commands to train, validate and evaluate multitask model on the challenge dataset. 
- Model training:
    ```python main.py --resume_model path/to/saved/model-weights```
- Model evaluation:
    ```python main.py --test_model --resume_model path/to/saved/model-weights```

The prediction file on `publictest.csv` will be generated for the post-evaluation phase of the competition.