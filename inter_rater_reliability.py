# This source code is based on the souce code of Nicholls and Bright
# which is available from http://dx.doi.org/10.5287/bodleian:R5qdeJxYA.

import csv
import numpy as np
import pandas as pd
import time

from krippendorff import alpha

mappings_file_a = 'hand_coded/story_pairs-2021-06-29-coder1-gedikli.csv'
mappings_file_b = 'hand_coded/story_pairs-2021-06-29-coder2-unaware_student.csv'

def mappings_from_file(fn):
    """
    Open manual mappings file and write results to list for IRR.

    """
    vals = []
    urls = set()

    # Get URLs
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            urls.update(set([str(line[0]),str(line[1])]))

    # Extract mappings from each file    
    with open(fn, 'r') as f:
        print('Extracting mappings from', fn)
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            # Skip those not in set
            if str(line[0]) not in urls or str(line[1]) not in urls:
                continue
            relation = str(line[2])
            relvalue = 0
            if relation == '1':
                relvalue = 1
            vals.append(relvalue)
    return vals

def get_mappings_array(fna, fnb):
    """
    Open an pair of CSVs with manual mappings data. Returns a numpy
    array suitable for feeding into alpha().

    """
    vals_a = mappings_from_file(fna)
    vals_b = mappings_from_file(fnb)
    
    print('Manual mappings a: {}/{} related.'.format(len([x for x in vals_a if x]),
                                                     len(vals_a)))
    print('Manual mappings b: {}/{} related.'.format(len([x for x in vals_b if x]),
                                                     len(vals_b)))
    
    assert len(vals_a) == len(vals_b)
    return np.array((vals_a, vals_b))

def show_diff(fna, fnb):
    dfa = pd.read_csv(fna, sep=';', names=['url1', 'url2', 'relation'], encoding='utf-8')
    dfb = pd.read_csv(fnb, sep=';', names=['url1', 'url2', 'relation'], encoding='utf-8')

    dfa_rows = {}
    for index, row in dfa.iterrows():
        dfa_rows[tuple(sorted([row[0], row[1]]))] = row[2]

    dfb_rows = {}
    for index, row in dfb.iterrows():
        dfb_rows[tuple(sorted([row[0], row[1]]))] = row[2]

    same_coded_counter = 0
    first_coder_0_and_second_coder_0 = 0
    first_coder_0_and_second_coder_1 = 0
    first_coder_1_and_second_coder_0 = 0
    first_coder_1_and_second_coder_1 = 0
    for index, row in dfa.iterrows():
        url1 = row[0]
        url2 = row[1]
        related1 = row[2]
        related2 = dfb_rows[tuple(sorted([url1, url2]))]
        if related1 == related2:
            same_coded_counter += 1
            if related1 == 1:
                first_coder_1_and_second_coder_1 += 1
            else:
                first_coder_0_and_second_coder_0 += 1
        else:
            print('Diff: {}, {}'.format(url1, url2))
            print('1st coder: {}, 2nd coder: {}'.format(related1, related2))
            if related1 == 1:
                first_coder_1_and_second_coder_0 += 1
            else:
                first_coder_0_and_second_coder_1 += 1

    not_same_coded = len(dfa) - same_coded_counter
    print('Matching rows: {}/{} = {}'.format(same_coded_counter, len(dfa), same_coded_counter/len(dfa)))
    print('Not matching rows: {}/{} = {}'.format(not_same_coded, len(dfa), not_same_coded/len(dfa)))
    print('First coder 0 and second coder 0: {}/{} = {}'.format(first_coder_0_and_second_coder_0, len(dfa), first_coder_0_and_second_coder_0 / len(dfa)))
    print('First coder 1 and second coder 1: {}/{} = {}'.format(first_coder_1_and_second_coder_1, len(dfa), first_coder_1_and_second_coder_1 / len(dfa)))
    print('First coder 1 and second coder 0: {}/{} = {}'.format(first_coder_1_and_second_coder_0, not_same_coded, first_coder_1_and_second_coder_0 / not_same_coded))
    print('First coder 0 and second coder 1: {}/{} = {}'.format(first_coder_0_and_second_coder_1, not_same_coded, first_coder_0_and_second_coder_1 / not_same_coded))

manual_mappings_full = get_mappings_array(mappings_file_a, mappings_file_b)

#####
#TEST: Calculate Krippendorff's alpha
#####

# Print setup info    
print('***** TEST RESULTS at', time.ctime())

print('Coder 1 Related/Total: {}/{}'.format(sum(manual_mappings_full[0]),
                                            len(manual_mappings_full[0])))
print('Coder 2 Related/Total: {}/{}'.format(sum(manual_mappings_full[1]),
                                            len(manual_mappings_full[1])))
print('Krippendorff\'s alpha: {}'.format(
        alpha(reliability_data=manual_mappings_full,
            level_of_measurement='nominal')))

show_diff(mappings_file_a, mappings_file_b)
