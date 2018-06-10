# Introduction to Natural Language Processing, SS18
## Word Vectors Project
This project concerns itself with Word Vectors on a Wikipedia dump corpus and their handling of ambiguous words.

We chose an [English Wikipedia Dump](https://dumps.wikimedia.org/enwiki/20180601/enwiki-20180601-pages-articles.xml.bz2) for training the Word Vectors.  
We picked two topics that share some vocabulary, with different semantics per topic. For the two topics, we used **Computer Science** and **Biology**.  
Between those topics, some ambiguous words are *tree*, *mouse* and *memory*.  


### Authors
* Maximilian Moser, 01326252
* Wolfgang Weintritt, 01327191


### Requirements
* `Python 3.6`
* `pipenv`


### Running it
* Place the downloaded Wikipedia dump in the `dumps/` directory
* `pipenv install`
* `pipenv run src/train.py -h`