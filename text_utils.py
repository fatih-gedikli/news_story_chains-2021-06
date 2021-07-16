import pandas as pd
import re
import Stemmer

from nltk.corpus import stopwords

stemmer_eng = Stemmer.Stemmer('english')
stopwords_eng = set(stopwords.words('english'))

# Use string.punctuation without - as template plus additional punctuation characters
punctuation_overall = '!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~' + '”'

# punctuation_overall without $ and +:
punctuation_left = punctuation_overall.replace('$', '').replace('+', '')

# punctuation_overall without $ and %:
punctuation_right = punctuation_overall.replace('$', '').replace('%', '')


########################################################
# Auxiliary methods for Named Entity Recognition (NER) #
########################################################

def clean_org(x):
    return clean_named_entities(x, sep=';', remove_apostrophes=True, replace_punctuation_by_whitespace=True)

def clean_per(x):
    return clean_named_entities(x, sep=';', remove_apostrophes=True, remove_punctuation_left=True, remove_punctuation_right=True)

def clean_loc(x):
    return clean_named_entities(x, sep=';', remove_apostrophes=True, replace_punctuation_by_whitespace=True)

def clean_named_entities(
    named_entities,
    sep=';',
    lower_case_non_acronyms=False,
    lower_case=False,
    remove_apostrophes=False,
    remove_duplicates=False,
    remove_punctuation_left=False,
    remove_punctuation_right=False,
    replace_punctuation_by_whitespace=False,
    remove_stopwords=False,
    stemming=False):

    if pd.isna(named_entities):
        return named_entities

    if not isinstance(named_entities, str):
        named_entities = str(named_entities)

    if lower_case:
        named_entities = named_entities.lower()

    if remove_apostrophes:
        named_entities = re.sub(r'(\w+)[^\w\s;]s', r'\1', named_entities)

    # Split named entities
    named_entities_list = named_entities.split(sep)

    if remove_duplicates:
        named_entities_list = remove_duplicates_from_list(named_entities_list)

    entities = []
    for e in named_entities_list:
        if not e:
            continue

        entity = e

        if lower_case_non_acronyms:
            if not entity.isupper():
                entity = entity.lower()

        if remove_punctuation_left:
            entity = entity.lstrip(punctuation_left)

        if remove_punctuation_right:
            entity = entity.rstrip(punctuation_right)

        if replace_punctuation_by_whitespace:
            # Replace all punctuation by whitespace
            entity = re.sub(r'[{}]'.format(punctuation_overall), ' ', entity)

        if remove_stopwords:
            new_entity = []
            for w in entity.split():
                if w not in stopwords_eng:
                    new_entity.append(w)
            entity = ' '.join(new_entity)

        if stemming:
            entity = ' '.join([stemmer_eng.stemWord(token) for token in entity.split()])

        # Remove superfluous spaces
        entity = ' '.join(entity.split())

        entities.append(entity)

    if remove_duplicates:
        entities = remove_duplicates_from_list(entities)

    return ';'.join(entities)

def remove_duplicates_from_list(str_list):
    """Removes duplicates from list and preserves order."""
    seen = set()
    result = []
    for item in str_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
