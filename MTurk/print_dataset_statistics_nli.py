import csv
import glob
import os

import pandas as pd


if __name__ == '__main__':

    root_dir = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )
    circa_split_files = glob.glob(f'{root_dir}/circa/data/circa-*.tsv')

    column_names = [
        'id',
        'context',
        'question_x',
        'canquestion_x',
        'answer_y',
        'judgements',
        'goldstandard1',
        'goldstandard2',
    ]

    df = {
        'split': [],
        'match_type': [],
        'seed': [],
        'label': [],
        'avg_chars': []
    }

    circa_nli_labels_relaxed = {
        'Yes': 'entailment',
        'Yes, subject to some conditions': 'none',
        'No': 'contradiction',
        'In the middle, neither yes nor no': 'neutral',
        'NA': 'none',
        'Other': 'none'
    }

    for split_file in circa_split_files:
        _, split_name, match_type, seed  = os.path.splitext(os.path.basename(split_file))[0].split('-')
        with open(split_file, 'r') as sf:
            tsv_reader = csv.DictReader(sf, delimiter='\t', fieldnames=column_names)
            next(tsv_reader)

            for line in tsv_reader:
                nli_label = circa_nli_labels_relaxed[line['goldstandard2']]
                num_characters = line['canquestion_x'] + line['answer_y']
                if match_type == 'unmatched':
                    num_characters += line["context"]
                num_characters = len(num_characters)

                df['match_type'].append(match_type)
                df['split'].append(split_name)
                df['seed'].append(seed)
                df['label'].append(nli_label)
                df['avg_chars'].append(num_characters)

    df = pd.DataFrame(df)
    pt = df.pivot_table(
        index=['match_type', 'split', 'seed'],
        columns='label',
        aggfunc='size'
    ).reset_index(-1)
    avg_chars = df.groupby(['match_type', 'split', 'seed'])['avg_chars'].mean().reset_index(-1)['avg_chars']
    pt = pd.concat([pt, avg_chars], axis=1)

    pt = pt.groupby(['match_type', 'split']).aggregate({
        'avg_chars': ['mean', 'std'],
        'entailment': ['mean', 'std'],
        'contradiction': ['mean', 'std'],
        'neutral': ['mean', 'std'],
        'none': ['mean', 'std'],
    }).reset_index()

    pt.columns =  ['match_type', 'split', *['_'.join((col[1], col[0])) for col in pt.columns[2:]]]

    pt['none'] = pt['mean_none'].round(2).astype(str) + ' $\pm$ ' + pt['std_none'].round(2).astype(str)
    pt['entailment'] = pt['mean_entailment'].round(2).astype(str) + ' $\pm$ ' + pt['std_entailment'].round(2).astype(str)
    pt['contradiction'] = pt['mean_contradiction'].round(2).astype(str) + ' $\pm$ ' + pt['std_contradiction'].round(2).astype(str)
    pt['neutral'] = pt['mean_neutral'].round(2).astype(str) + ' $\pm$ ' + pt['std_neutral'].round(2).astype(str)
    pt['avg_chars'] = pt['mean_avg_chars'].round(2).astype(str) + ' $\pm$ ' + pt['std_avg_chars'].round(2).astype(str)

    pt = pt.drop(columns=[
        'mean_none',
        'std_none',
        'mean_entailment',
        'std_entailment',
        'mean_contradiction', 
        'std_contradiction', 
        'mean_neutral', 
        'std_neutral', 
        'mean_avg_chars', 
        'std_avg_chars'
    ])

    pt = pt.sort_values(by=['match_type', 'split'])

    print(pt.to_latex(index=False, escape=False))
