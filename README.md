# Replication materials for the paper "Automated Identification of News Story Chains: A New Dataset and Entity-based Labeling Method" by Gedikli, Stockem-Novo, and Jannach

This repository contains code and data underlying the paper.

## Setup

1. `conda create -n nsc python=3.8.5`
2. `conda install nltk numpy pandas`
3. `conda install pytorch cudatoolkit=10.2 -c pytorch`
4. `pip install krippendorff pymongo[srv] PyStemmer simpletransformers`

## Content

* Directory `hand_coded` contains the hand-codings of both coders and the file `diff_coder1_coder2.txt` which contains a description of the errors for each article pair.
* `dataset_generator.py` contains our main article selection procedure.
* `inter_rater_reliability.py` contains a script for computing the inter-rater reliability between both coders by computing the Krippendorff’s alpha score.
* `nicholls_and_bright_dataset_analysis.py` contains a script for annotating the dataset of Nicholls & Bright with named entities and for computing the common named entities of related articles. The script identifies all related tuples with no common entities.
* `text_utils.py` contains auxiliary methods.
* `gedikli-business_energy_news_dataset-2021-06-29-article_texts.csv` contains the extracted texts and the named entities of the 100 news articles from the Business Energy News dataset.
* `gedikli-business_energy_news_dataset-2021-06-29.[csv|xlsx]` contains our new dataset of Business Energy News which was created with the help of our method.
