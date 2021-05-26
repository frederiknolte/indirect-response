import os
from itertools import combinations

import numpy as np
import pandas as pd
import sacrebleu
from scipy.stats import wilcoxon
from tqdm import tqdm

CIRCA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../circa/circa-data.tsv"
)


def read_circa(tsv_file: os.PathLike = CIRCA) -> pd.DataFrame:

    df = pd.read_csv(CIRCA, delimiter="\t")
    for col in df.columns:
        if col == "id":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype("string")

    df = df[~df["canquestion-X"].isna()]
    return df


def _calc_bleu(_df):

    bleu = sacrebleu.corpus_bleu(
        _df["answer-Y"].tolist(), [_df["canquestion-X"].tolist()]
    )
    return bleu


def calc_bleu_scores(tsv_file=CIRCA):
    df = read_circa(tsv_file)

    bleu = _calc_bleu(df)
    bleu_gs1 = df.groupby("goldstandard1").apply(_calc_bleu)
    bleu_gs2 = df.groupby("goldstandard2").apply(_calc_bleu)

    return bleu, bleu_gs1, bleu_gs2


def bleu_significance_test(
    tsv_file=CIRCA, relaxed=True, num_samples=100, sample_size=100
):
    '''wilcoxon test for samples from pairs of categories'''

    df = read_circa(tsv_file)
    if relaxed:
        goldstandard = "goldstandard2"
    else:
        goldstandard = "goldstandard1"
    df = df[~df[goldstandard].isna()]
    labels = df[goldstandard].unique()

    significance = dict()
    for l1, l2 in tqdm(combinations(labels, 2)):
        bleu1 = np.zeros(num_samples)
        bleu2 = np.zeros(num_samples)
        for ct in range(num_samples):
            bleu1[ct] = _calc_bleu(
                df[df[goldstandard] == l1].sample(sample_size, replace=True)
            ).score
            bleu2[ct] = _calc_bleu(
                df[df[goldstandard] == l2].sample(sample_size, replace=True)
            ).score
        significance[f"{l1} - {l2}"] = wilcoxon(bleu1, bleu2)

    return significance


if __name__ == "__main__":
    df = read_circa()
    bleu, bleu_gs1, bleu_gs2 = calc_bleu_scores()
    significance = bleu_significance_test()
