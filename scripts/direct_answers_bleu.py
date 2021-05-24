import os

import numpy as np
import pandas as pd
import sacrebleu

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

    return df


def calc_bleu_scores(tsv_file=CIRCA):
    df = read_circa(tsv_file)

    bleu = sacrebleu.corpus_bleu(
        df["answer-Y"].tolist(), [df["canquestion-X"].tolist()]
    )
    bleu_gs1 = df.groupby("goldstandard1").apply(
        lambda _df: sacrebleu.corpus_bleu(
            _df["answer-Y"].tolist(), [_df["canquestion-X"].tolist()]
        )
    )
    bleu_gs2 = df.groupby("goldstandard2").apply(
        lambda _df: sacrebleu.corpus_bleu(
            _df["answer-Y"].tolist(), [_df["canquestion-X"].tolist()]
        )
    )

    print("bleu:", bleu, "per goldstandard1:", bleu_gs1, "per goldstandard2:", bleu_gs2)
    return bleu, bleu_gs1, bleu_gs2


if __name__ == "__main__":
    df = read_circa()
    calc_bleu_scores()
