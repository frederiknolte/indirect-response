import argparse
import pandas
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from krippendorff_alpha import krippendorff_alpha, nominal_metric, interval_metric


def analysis(input_csv, result_csv):
    analysis_str = ""
    delimiter = "\n\n============================================================\n\n"

    #### PREPARATION ####

    # Read the data
    inputs = pd.read_csv(input_csv)
    results = pd.read_csv(result_csv, sep=',', dtype=object, engine='python',
                          usecols=['WorkerId',
                                   'Input.context',
                                   'Input.question',
                                   'Input.answer',
                                   'Input.explanation',
                                   'Input.interpretation',
                                   'Answer.explanation-quality.label']
                          )
    results.columns = ['WorkerId', 'context', 'question',
                       'answer', 'explanation', 'shown_target',
                       'annotation']

    # Clean columns
    results['annotation'].replace('5 - very good', 5, inplace=True)
    results['annotation'].replace('1 - very poor', 1, inplace=True)
    results.annotation = results.annotation.astype(int)

    # Match dataframes
    data = inputs.merge(
            results,
            on=["context", "question", "answer", "explanation"],
            how="right",
    )
    data['corr_target_shown'] = data['target'] == data['shown_target']
    print(data.columns)

    #### ANALYSES ####


    #### Without normalization ####

    # Simple aggregations of user annotations
    agg_data = data.groupby(['leaked', 'correct_pred', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['leaked']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by leakage:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['shown_target', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by shown target:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['target', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by true target:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    # Faithfulness
    agg_data = data[data["prediction"] == data["shown_target"]].groupby(['leaked', 'correct_pred', 'corr_target_shown']).agg({'annotation': ['mean','std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Faithfulness Aggregation (prediction == shown target):\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter


    #### Worker Analysis ####

    mean_buckets = np.linspace(0, 5, 11)
    agg_data = data.groupby(['WorkerId']).agg({'annotation': ['mean', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    # print(agg_data)


    # Krippendorff's alpha
    agg_data = data.copy()
    agg_data['dummy_id'] = agg_data['id'].astype(str) + agg_data['corr_target_shown'].astype(str)
    agg_data = agg_data.pivot(index='WorkerId', columns='dummy_id', values='annotation')
    agg_data = agg_data.fillna('*')
    ka_nominal = krippendorff_alpha(agg_data.to_numpy(), nominal_metric, missing_items='*')
    ka_interval = krippendorff_alpha(agg_data.to_numpy(), interval_metric, missing_items='*')
    print(f"Krippendorff\'s alpha nominal: {ka_nominal}")
    print(f"Krippendorff\'s alpha interval: {ka_interval}")


    #### With worker normalization ####  # TODO is this even correct? Check tomorrow
    analysis_str += "#########################\nWith normalization:\n#########################\n\n"

    annotation_mean = data.groupby(['WorkerId']).agg({'annotation': ['mean']}).mean().to_numpy()[0]
    annotation_std = data.groupby(['WorkerId']).agg({'annotation': ['std']}).mean().to_numpy()[0]  # TODO what to do with this

    agg_data = data.groupby(['WorkerId']).agg({'annotation': ['mean']})
    agg_data.columns = ['worker_mean']
    agg_data['mean_diff'] = annotation_mean - agg_data['worker_mean']
    agg_data = data.merge(agg_data, how='left', on=['WorkerId'])
    agg_data['centered_annotation'] = agg_data['annotation'] + agg_data['mean_diff']  # TODO what to do now with ratings outside [1,5]?

    # Simple aggregations of user annotations
    agg_data = data.groupby(['leaked', 'correct_pred', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['leaked']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by leakage:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['shown_target', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by shown target:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    agg_data = data.groupby(['target', 'corr_target_shown']).agg({'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Simple Aggregation by true target:\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    # Faithfulness
    agg_data = data[data["prediction"] == data["shown_target"]].groupby(['leaked', 'correct_pred', 'corr_target_shown']).agg(
        {'annotation': ['mean', 'std', 'count']})
    agg_data.columns = agg_data.columns.to_flat_index()
    analysis_str += "Faithfulness Aggregation (prediction == shown target):\n\n"
    analysis_str += agg_data.to_string()
    analysis_str += delimiter

    print(analysis_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="input csv file from LAS")
    parser.add_argument("--result_csv", help="results csv file from LAS")
    args = parser.parse_args()

    analysis(args.input_csv, args.result_csv)
