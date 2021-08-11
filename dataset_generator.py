# Article Selection Procedure
# ---------------------------------------------------------
# 1. Cluster articles according to NESM similarity metric
# 2. Repeat until n articles found:
#    2.1 Randomly select a cluster
#    2.2 Append articles from cluster to dataset

import csv
import os
import pandas as pd
import pickle
import pymongo
import random

from pathlib import Path
from urllib.parse import urlparse

mongo_username = 'USERNAME'
mongo_password = 'PASSWORD'

mongo_client = pymongo.MongoClient('mongodb+srv://{}:{}@...'.format(mongo_username, mongo_password))
mongo_client_test = mongo_client.test
database = mongo_client['api_dev']
news_collection = database['news']

output_dir = './story_chain_detection/'
random_seed = 1453


class NESM():
    def __init__(self):
        fname = os.path.join(output_dir, 'sim.pickle')
        if Path(fname).is_file():
            # Load scores (number of common entities between two documents) and
            # maximum number of common entities per category from cache.
            self.cache_score = pickle.load(open(fname, 'rb'))
            print('Scores unpickled and loaded')
            return
        
        print('Compute max number of common orgs, pers and locs...')
        self.cache_score = {
            'max_org': 0,
            'max_per': 0,
            'max_loc': 0,
            'org': {},
            'per': {},
            'loc': {}
        }

        # Load business energy news articles from database.
        news = []
        cursor = news_collection.find({'classification.energy': True}, batch_size=100)
        counter = 0
        for article in cursor:
            if counter % 10 == 0:
                print(counter)
            counter += 1
            light_article = {
                'url': article['url'],
                'org': article['org'],
                'per': article['per'],
                'loc': article['loc'],
            }
            news.append(light_article)

        print('Number of articles:', len(news))

        # Compute scores (number of common entities between two documents) and
        # maximum number of common entities per category.
        counter = 0
        for article1 in news:
            article1_passed = False
            for article2 in news:
                if not article1_passed:
                    if article1['url'] == article2['url']:
                        article1_passed = True
                    continue
                num_common_orgs = self.score('org', article1, article2)
                num_common_pers = self.score('per', article1, article2)
                num_common_locs = self.score('loc', article1, article2)
                if num_common_orgs > self.cache_score['max_org']:
                    self.cache_score['max_org'] = num_common_orgs
                if num_common_pers > self.cache_score['max_per']:
                    self.cache_score['max_per'] = num_common_pers
                if num_common_locs > self.cache_score['max_loc']:
                    self.cache_score['max_loc'] = num_common_locs

            counter += 1
            if counter % 10 == 0:
                print('Processed:', counter)

        print('Max number of common orgs:', self.cache_score['max_org'])
        print('Max number of common pers:', self.cache_score['max_per'])
        print('Max number of common locs:', self.cache_score['max_loc'])

        # Save scores (number of common entities between two documents) and
        # maximum number of common entities per category in the cache.
        pickle.dump(self.cache_score, open(fname, 'wb'))
        print('Scores computed and pickled')

    def score(self, cat, article1, article2):
        try:
            return self._getcachedscore(cat, article1['url'], article2['url'])
        except KeyError:
            return self._rescore(cat, article1, article2)

    def _rescore(self, cat, article1, article2):
        score = NESM.get_num_of_common_ne(cat, article1, article2)
        self._setcachedscore(cat, article1['url'], article2['url'], score)
        return score

    def _getcachedscore(self, cat, url1, url2):
        return self.cache_score[cat][tuple(sorted([url1, url2]))]

    def _setcachedscore(self, cat, url1, url2, score):
        self.cache_score[cat][tuple(sorted([url1, url2]))] = score

    def named_entities_shared_measure(self, article1, article2):
        url1 = article1['url']
        url2 = article2['url']

        num_common_orgs = self.cache_score['org'][tuple(sorted([url1, url2]))]
        num_common_pers = self.cache_score['per'][tuple(sorted([url1, url2]))]
        num_common_locs = self.cache_score['loc'][tuple(sorted([url1, url2]))]

        return (num_common_orgs/self.cache_score['max_org']) + (num_common_pers/self.cache_score['max_per']) + (num_common_locs/self.cache_score['max_loc'])

    @staticmethod
    def get_common_ne(cat, article1, article2):
        ne1 = []
        ne2 = []
        if type(article1[cat]) == list:
            # When we load the data from the database, we have to parse a list structure.
            ne1 = [entity['name'] for entity in article1[cat]]
            ne2 = [entity['name'] for entity in article2[cat]]
        else:
            if type(article1[cat]) == str:
                ne1 = article1[cat].split(';')
            if type(article2[cat]) == str:
                ne2 = article2[cat].split(';')
        return list(set(ne1).intersection(set(ne2)))

    @staticmethod
    def get_num_of_common_ne(cat, article1, article2):
        return len(NESM.get_common_ne(cat, article1, article2))    


def cluster_articles(sim, sim_score_cutoff):
    """
    Clusters news articles according to a given similarity metric

    Parameters:
    sim (function): Similarity metric for comparing two articles
    sim_score_cutoff (float): Minimum similarity score to accept as a cluster match

    Returns:
    List of article clusters ([ [a11, a12, ...], [a21, a22, ...], ... ])
    """
    fname = os.path.join(output_dir, 'cluster.pickle')
    if Path(fname).is_file():
            # Load clusters from cache.
            clusters = pickle.load(open(fname, 'rb'))
            print('Clusters unpickled and loaded')
            return clusters

    clusters = []
    cursor = news_collection.find({'classification.energy': True}, batch_size=100)
    for article in cursor:
        sim_max = 0
        sim_max_c = None
        for c in clusters:
            for a in c:
                sim_tmp = sim.named_entities_shared_measure(article, a)
                if sim_tmp > sim_max:
                    sim_max = sim_tmp
                    sim_max_c = c
        
        if sim_max >= sim_score_cutoff:
            # Add article to cluster.
            sim_max_c.append(article)
        else:
            # Create new cluster and add article to newly created cluster.
            clusters.append([article])

    # Save clusters in the cache.
    pickle.dump(clusters, open(fname, 'wb'))
    print('Clusters computed and pickled')
    return clusters

def select_articles(n, clusters, exclude_clusters_with_articles=[], min_cluster_size=2, max_articles_from_cluster=30):
    """
    Article Selection Procedure

    Returns:
    List of n articles ([a1, a2, ..., an])
    """
    res = []
    random.Random(random_seed).shuffle(clusters)
    for c in clusters:
        if len(c) < min_cluster_size:
            continue
        exclude_cluster = False
        if exclude_clusters_with_articles:
            for a in c:
                if exclude_cluster:
                    break
                for exclude_topic in exclude_clusters_with_articles:
                    if exclude_topic.lower() in a['headline'].lower():
                        exclude_cluster = True
                        break
        if exclude_cluster:
            continue
        random.Random(random_seed).shuffle(c)
        num_articles_from_cluster = 0
        for a in c:
            if num_articles_from_cluster < max_articles_from_cluster:
                # Skip duplicate articles. We consider an item to be a duplicate
                # if there is already a news item in the result record with an 80% match in the title.
                is_duplicate = False
                for res_a in res:
                    if res_a['headline'] == a['headline'] or res_a['url'] == a['url'] or (
                      (len(list(set(res_a['headline'].split()).intersection(set(a['headline'].split())))) / len(a['headline'].split())) > 0.8):
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                print('Picking one from cluster of length {}: {}'.format(len(c), a['headline']))
                res.append(a)
                num_articles_from_cluster += 1
            else:
                break
            if len(res) == n:
                df = pd.DataFrame(columns=['url', 'paperurl', 'title', 'date', 'text', 'summary', 'org', 'per', 'loc'])
                for a in res:
                    parsed_url = urlparse(a['url'])
                    paper_url = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url)
                    df = df.append({
                        'url': a['url'],
                        'paperurl': paper_url,
                        'title': a['headline'],
                        'date': a['datePublished'],
                        'text': a['abstract'] + ' ' + a['articleBody'],
                        'summary': a['abstract'],
                        'org': [entity['name'] for entity in a['org']],
                        'per': [entity['name'] for entity in a['per']],
                        'loc': [entity['name'] for entity in a['loc']],
                    }, ignore_index=True)
                return res
        
        print('---')

def write_dataset_to_file(articles):
    """
    Bootstrap the dataset with the given n articles, i.e., generate all ((n * (n-1)) / 2) separate pairwise comparisons.
    """
    df = pd.DataFrame(columns=['url1', 'url2', 'relation', 'title1', 'title2', 'num_common_ne', 'common_org', 'common_per', 'common_loc'])
    for a in articles:
        a_passed = False
        for b in articles:
            if not a_passed:
                if a['url'] == b['url']:
                    a_passed = True
                continue
            common_org = NESM.get_common_ne('org', a, b)
            common_per = NESM.get_common_ne('per', a, b)
            common_loc = NESM.get_common_ne('loc', a, b)
            df = df.append({
                'url1': a['url'],
                'url2': b['url'],
                'relation': '?',
                'title1': a['headline'] if 'headline' in a else a['title'],
#                'summary1': a['abstract'],
                'title2': b['headline'] if 'headline' in b else b['title'],
#                'summary2': b['abstract'],
                'num_common_ne': len(common_org) + len(common_per) + len(common_loc),
                'common_org': common_org,
                'common_per': common_per,
                'common_loc': common_loc,
            }, ignore_index=True)
    
    print(df)

    df.sort_values(['url1', 'num_common_ne'], ascending = (True, False))
    df.to_csv(os.path.join(output_dir, 'story_chain.csv'), sep=',', quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8', header=True, index=False)

def remove_blacklist_ne(named_entities, sep=';', blacklist=[]):
    if pd.isna(named_entities):
        return named_entities

    if not isinstance(named_entities, str):
        named_entities = str(named_entities)

    # Split named entities
    named_entities_list = named_entities.split(sep)

    entities = []
    for e in named_entities_list:
        if not e:
            continue
        if not e in blacklist:
            entities.append(e)

    return sep.join(entities)

def main():
    verbose = False
    cluster_articles_for_selection = False
    fname = os.path.join(output_dir, 'nicholls_and_bright_dataset_with_ner.csv')
    df = pd.DataFrame()
    if Path(fname).is_file():
        df = pd.read_csv(fname, sep=',', quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8')
        ner_blacklist = ['Guardian', 'Mirror Online', 'BBC', 'BBC News', 'ABC News', 'Mail Online', 'The Mail', 'Associated Newspapers LtdPart', 'Daily Express', 'Online', 'Internet Explorer', 'JavaScript', 'Twitter', 'Facebook', 'News', 'The', 's', 'A', 'Sunday', ]
        df['org'] = df['org'].apply(lambda x: remove_blacklist_ne(x, blacklist=ner_blacklist))
        df['per'] = df['per'].apply(lambda x: remove_blacklist_ne(x, blacklist=ner_blacklist))
        df['loc'] = df['loc'].apply(lambda x: remove_blacklist_ne(x, blacklist=ner_blacklist))
        print(df)

    if cluster_articles_for_selection:
        sim = NESM()
        clusters = cluster_articles(sim, 0.4)
        clusters.sort(key = len)

        print('-' * 50)
        print('Number of clusters found:', len(clusters))

        if verbose:
            print('-' * 50)
            print('Article distribution in clusters:')
            for i, c in enumerate(clusters):
                if len(c) > 2:
                    print('#' * 50)
                    print('Cluster {}: {}'.format(i, len(c)))
                    titles = []
                    for a in c:
                        titles.append(a['headline'])
                    print('\n'.join(titles))

        print('-' * 50)
        # Exclude all clusters dealing with stock markets as the articles 
        # in these clusters were very similar and very likely to be generated automatically.
        exclude_clusters_with_articles = [
          'Oil',
          'Price',
          'Stake',
          'Stock',
          'Update'
        ]
        articles = select_articles(100, clusters, exclude_clusters_with_articles, min_cluster_size=2, max_articles_from_cluster=30)
    else:
        articles = []
        for index, a in df.iterrows():
            articles.append(a)

    articles_df = pd.DataFrame(articles)
    articles_df.to_csv(os.path.join(output_dir, 'story_chain_news_articles.csv'), sep=',', quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8', header=True, index=False)
    write_dataset_to_file(articles)

if __name__ == '__main__':
    main()
