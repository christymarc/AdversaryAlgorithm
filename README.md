# CS159 Final Project

Final project for NLP Spring 2023. Implementing PWWS algorithm for adversarial example generation of DNN sentiment analysis models.

## Notes
PWWS algorithm inspired by this paper [Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency](https://aclanthology.org/P19-1103.pdf) by Ren, et al. DNNs trained and implemented through fastai, [the IMDB movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The PWWS algorithm creates semantically and syntactically sensical adversarial examples for sentences of text through replacing words with their synonyms and named entities (NEs) with similar NEs. The algorithm used word saliency (the degree of change in the output probability if a word is set to unknown) and the change in classification probability after word replacement in order to score that replacement. Then the scores of each replacement are sorted by strength and the strongest replacements to the weakest replacements are applied to the sentence in order until the sentence's classification is altered (if it is). What is yielded from this process is a sensical adversarial example that reveals a weakness of the network.

## Code Overview
The code for training the models, the PWWS Algorithm, and data processing/visualization is in the code folder. 
* train_model.py and run_training.py: trains the DNNs
* adversary.py and run_adversary.py: implements + run the PWWS algorithm
  * NE_info.py: adds support in the algorithm for named entities
* model_training.ipynb and adversary.ipynb: notebooks used in the process of model and PWWS algorithm development (notebooks for exploration/testing/debugging)
* data.ipynb: notebook used for looking at collected data and visualization
