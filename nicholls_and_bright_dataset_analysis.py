import csv
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from simpletransformers.ner import NERModel
from text_utils import clean_org, clean_per, clean_loc
from urllib.parse import urljoin

hand_coded_dir = './hand_coded/'
file_coder2 = 'story_pairs-2021-06-29-coder2-unaware_student.csv'
input_dir = './input/nicholls_and_bright/'
file_list = ['bbc-out.csv', 'express-out.csv', 'guardian-out.csv', 'mail-out.csv', 'mirror-out.csv']
validation_dataset = 'story_pairs_validation_august.csv'
output_dir = './story_chain_detection/'
path_to_ner_model = './machine_learning/ner/models/best_model'

def section_tokenize(news_article, max_seq_length=512):
    ret = []
    sent_tokenized = sent_tokenize(news_article)

    i = 0
    while i < len(sent_tokenized):
        section = sent_tokenized[i]
        i = i + 1
        for j in range(i, len(sent_tokenized)):
            if len(section) + len(sent_tokenized[j]) < max_seq_length:
                # Extend section
                i = j + 1
                section += ' ' + sent_tokenized[j]
            else:
                i = j
                break
        ret.append(section)

    return ret

def get_org(prediction):
    return get_entities(prediction, beginning_tag='B-ORG', inside_tag='I-ORG')

def get_per(prediction):
    return get_entities(prediction, beginning_tag='B-PER', inside_tag='I-PER')

def get_loc(prediction):
    return get_entities(prediction, beginning_tag='B-LOC', inside_tag='I-LOC')

def get_entities(prediction, beginning_tag='B-ORG', inside_tag='I-ORG'):
    entities = []
    entity = ''
    for section in range(len(prediction)):
        # mapping: word -> tag, e.g.: Argentina -> B-LOC
        for mapping in prediction[section]:
            for key in mapping:
                if mapping[key] == beginning_tag:
                    entity = key
                elif mapping[key] == inside_tag:
                    entity = entity + ' ' + key if entity else key
                else:
                    new_entity = entity.strip().rstrip(',')
                    if new_entity:
                        entities.append(new_entity)
                    entity = ''
    return ';'.join(entities)

def annotate_dataset():
    val_df = pd.read_csv(urljoin(input_dir, validation_dataset), sep=';', names=['Link1', 'Link2', 'Relation'], encoding='utf-8')
    #print(val_df)
    column_values = val_df[['Link1', 'Link2']].values.ravel()
    val_unique_links =  pd.unique(column_values)
    print(len(val_unique_links), 'unique links in validation dataset') # 254 unique links in validation dataset

    combined_df = pd.concat([pd.read_csv(urljoin(input_dir, fname), sep=',', names=['url', 'paperurl', 'title', 'date', 'text'], encoding='utf-8') for fname in file_list], ignore_index=True)
    # Filter articles which are not in validation dataset
    articles = []
    for index, article in combined_df.iterrows():
        if article['url'] in val_unique_links:
            articles.append(article)
    
    val_text_df = pd.DataFrame(articles, columns=['url', 'paperurl', 'title', 'date', 'text'], index=None)
    val_text_df.index = np.arange(0, len(val_text_df))
    print(val_text_df)

    model_input = []
    for index, article in val_text_df.iterrows():
        news_article = article['title'] + '. ' + article['text']
        section_tokenized = section_tokenize(news_article)
        model_input.append(section_tokenized)
    
    flat_model_input = [item for sublist in model_input for item in sublist]
    len_articles = [len(sublist) for sublist in model_input]
    assert len(len_articles) == len(val_text_df.index)

    predictions = []
    if model_input:
        model = NERModel('roberta', path_to_ner_model)

        predictions, raw_outputs = model.predict(flat_model_input)
        assert len(predictions) == len(flat_model_input)

    orgs = []
    pers = []
    locs = []
    next_article_start_idx = 0
    for i, row in val_text_df.iterrows():
        next_article_end_idx = next_article_start_idx + len_articles[i]
        prediction_i = predictions[next_article_start_idx:next_article_end_idx]
        org = get_org(prediction_i)
        per = get_per(prediction_i)
        loc = get_loc(prediction_i)
        orgs.append(org)
        pers.append(per)
        locs.append(loc)
        next_article_start_idx = next_article_end_idx
    
    val_text_df['org'] = orgs
    val_text_df['per'] = pers
    val_text_df['loc'] = locs

    val_text_df['org'] = val_text_df['org'].apply(lambda x: clean_org(x))
    val_text_df['per'] = val_text_df['per'].apply(lambda x: clean_per(x))
    val_text_df['loc'] = val_text_df['loc'].apply(lambda x: clean_loc(x))

    print(val_text_df)
    val_text_df.to_csv(urljoin(output_dir, 'nicholls_and_bright_dataset_with_ner.csv'), sep=',', quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8', header=True, index=False)

def check_common_ne_of_related_articles():
    val_df = pd.read_csv(urljoin(input_dir, validation_dataset), sep=';', names=['Link1', 'Link2', 'Relation'], encoding='utf-8')
    val_df = val_df.loc[val_df['Relation'] == 'Related']
    print(val_df)

    val_ner_df = pd.read_csv(urljoin(output_dir, 'story_chain.csv'), sep=',', quoting=csv.QUOTE_ALL, quotechar='"', encoding='utf-8')

    ret_df = pd.DataFrame()
    for i, row in val_df.iterrows():
        link1 = row['Link1']
        link2 = row['Link2']
        ret = val_ner_df.loc[((val_ner_df['url1'] == link1) & (val_ner_df['url2'] == link2)) | ((val_ner_df['url1'] == link2) & (val_ner_df['url2'] == link1))]
        ret_df = ret_df.append(ret)
    
    print(ret_df)
    print(ret_df.describe())
    print('Related tuples with no common entities:')
    for i, row in ret_df.iterrows():
        if row['num_common_ne'] == 0:
            print('-'*50)
            print('url1: {}'.format(row['url1']))
            print('url2: {}'.format(row['url2']))
            print('title1: {}'.format(row['title1']))
            print('title2: {}'.format(row['title2']))

def compute_accuracy_of_auto_labeling_procedure():
    df = pd.read_csv('gedikli-business_energy_news_dataset-2021-06-29.csv', sep=';', encoding='utf-8')
    df = df.loc[df['num_common_ne'] == 0]
    print('Number of article pairs with no common named entitis that were automatically considered as unrelated:', len(df))

    coder2_df = pd.read_csv(urljoin(hand_coded_dir, file_coder2), sep=';', names=['Link1', 'Link2', 'Relation'], encoding='utf-8')
    print('Compute diffs')
    count_diffs = 0
    for i, row in df.iterrows():
        url1 = row['url1']
        url2 = row['url2']
        article_pair_coder2 = coder2_df.loc[((coder2_df['Link1'] == url1) & (coder2_df['Link2'] == url2)) | ((coder2_df['Link2'] == url1) & (coder2_df['Link1'] == url2))]
        assert len(article_pair_coder2) == 1
        if article_pair_coder2.iloc[0]['Relation'] != 0:
            count_diffs += 1
            print('-'*50)
            print('url1:', url1)
            print('url2:', url2)
            print('title1: {}'.format(row['title1']))
            print('title2: {}'.format(row['title2']))
    
    print('='*50)
    print('Number of diffs:', count_diffs)
    print('Precision:', (len(df)-count_diffs) / len(df))

def main():
    #annotate_dataset()
    #check_common_ne_of_related_articles()
    compute_accuracy_of_auto_labeling_procedure()

if __name__ == '__main__':
    main()
