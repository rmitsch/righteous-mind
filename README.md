# righteous-mind
Approximation of a empirical verification of the moral-based framework to distuingish between different political preferences as utilized in "The Righteous Mind" by Jonathan Haidt (2012).

## Alternatives for Semantic Encoding
* GuidedLDA
    * Con: No semantic context.
* ELMo w/ baseline sentence (SIF) encoding.
    * Con: Separate phrase encoding necessary.
* ELMo w/ word-to-phrase comparison.
    * Con: Separate phrase encoding necessary (for moral phrases).
    * Con: Number of comparisons skyrockets (n_words_in_tweets * n_moral_phrases instead of n_moral_phrases * n_tweets). 
* Universal Sentence Encoder.
    * Con: Apparently problem w/ handling OOV words.
* BERT
    * Con: Resource requirements apparently exceed machine capability.
    
    
------

#### Notes

Starting BERT server for moral matrix (i. e. we want one embedding vector for the whole phrase): 

```bert-serving-start -model_dir ~/Development/data/BERT/base/ -num_worker=1 -cpu``` 
 
Starting BERT server for tweets (i. e. we want one embedding vector per token): 

```bert-serving-start -model_dir ~/Development/data/BERT/base/ -num_worker=1 -cpu -pooling_strategy NONE``` 

----

#### Literature

- https://arxiv.org/abs/1709.05467
- https://www.researchgate.net/publication/258698999_Measuring_Moral_Rhetoric_in_Text
- http://morteza-dehghani.net/wp-content/uploads/morality-lines-detecting.pdf
- https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a
- https://arxiv.org/abs/1810.04805
- ElMo, Universal Sentence Encoder, SIF, Guided Topic Modeling