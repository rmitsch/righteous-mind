### Politics, Morals and Tweets: Identifying Political Affiliation Utilizing Moral Foundations Theory and Contextual Embeddings

This project seeks to extract moral preferences from US-American politician's tweets and predict their political ideology based on these moral preferences. The underlying theoretical framework is the _[Moral Foundations Theory](https://en.wikipedia.org/wiki/Moral_foundations_theory)_.
   
------

#### Notes

* Dependencies can be found in `environment.yml`.
* Starting BERT server for inferring seed phrases for moral values (i. e. we want one embedding vector for the whole phrase): 
```bert-serving-start -model_dir ~/Development/data/BERT/base/ -num_worker=1 -cpu``` 
* Starting BERT server for inferring words in tweets (i. e. we want one embedding vector per token): 
```bert-serving-start -model_dir ~/Development/data/BERT/base/ -num_worker=1 -cpu -max_seq_len=40 -pooling_strategy NONE``` 
* Created for the course "Social Media Data: Quantitative Text Analysis of Big Data", University of Vienna, Winter Semester 2018/2019.
