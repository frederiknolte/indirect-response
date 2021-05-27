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

GENERIC_ANSWERS_POSITIVE = [
    "sounds good",
    "sounds great",
    "good idea",
    "great idea",
    "let's do it",
    "I think so",
    "I'd love to",
]
GENERIC_ANSWERS_NEGATIVE = [
    "not anymore",
    "not really",
    "not yet",
    "I'm not a fan",
    "I have not",
]


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

    num_entries = _df.shape[0]
    generic_responses_padded = np.array(
        [GENERIC_ANSWERS_POSITIVE + GENERIC_ANSWERS_NEGATIVE] * num_entries
    ).T
    ref = [_df["canquestion-X"].tolist()] + generic_responses_padded.tolist()

    bleu = sacrebleu.corpus_bleu(_df["answer-Y"].tolist(), ref)
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
    """wilcoxon test for samples from pairs of categories"""

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


def calc_las_per_target(las_csv: os.PathLike, seed: int = 948):
    LABEL_MAP = {0: "neutral", 1: "entailment", 2: "contradiction", 3: "none"}

    df = pd.read_csv(las_csv)
    df["x"] = df[f"preds_circa_NLI_distilbert-base-cased_sim.ST.RE_seed{seed}_X"]
    df["xe"] = df[f"preds_circa_NLI_distilbert-base-cased_sim.ST.RE_seed{seed}_XE"]

    def _calc_las(df):
        leaked = df[df.leaked == "Yes"]
        l0 = (leaked.xe == leaked.prediction).mean() - (
            leaked.x == leaked.prediction
        ).mean()
        nonleaked = df[df.leaked != "Yes"]
        l1 = (nonleaked.xe == nonleaked.prediction).mean() - (
            nonleaked.x == nonleaked.prediction
        ).mean()
        las = (l0 + l1) / 2
        return las

    las = _calc_las(df)

    las_per_target = df.groupby("target").apply(_calc_las)
    las_per_target = {LABEL_MAP[k]: v for k, v in las_per_target.items()}

    return las, las_per_target


if __name__ == "__main__":
    df = read_circa()
    bleu, bleu_gs1, bleu_gs2 = calc_bleu_scores()
    significance = bleu_significance_test()
