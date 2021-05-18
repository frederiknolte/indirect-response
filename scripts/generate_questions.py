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


def gen_googleform_interpret(
    use_context: bool = True,
    strict: bool = True,
    num_questions: int = 20,
    tsv_file: os.PathLike = CIRCA,
    seed: int = 7,
) -> pd.DataFrame:
    """generate questions about interpretation of indirect answers"""

    df = read_circa(tsv_file)
    np.random.seed(seed)

    if strict:
        goldstandard = "goldstandard1"
    else:
        goldstandard = "goldstandard2"

    df = df[(df[goldstandard] != "nan") & (df[goldstandard] != "Other")]
    df_sampled = df.sample(num_questions)
    df_sampled.rename(columns={goldstandard: "goldstandard"}, inplace=True)

    df_sampled["help_text"] = (
        "QUESTION(X): "
        + df_sampled["question-X"]
        + "\t QUESTION(Y): "
        + df_sampled["answer-Y"]
    )
    if use_context:
        df_sampled["help_text"] = (
            "CONTEXT: " + df_sampled["context"] + "\t " + df_sampled["help_text"]
        )

    all_answers = df[goldstandard].unique().tolist()
    df_sampled["other_choices"] = df_sampled.goldstandard.apply(
        lambda x: ";".join([a for a in all_answers if x != a])
    )

    return df_sampled[["help_text", "goldstandard", "other_choices"]]


def gen_googleform_explain_nli(
    inputs_file: os.PathLike,
    targets_file: os.PathLike,
    predictions_file: os.PathLike,
    use_context: bool = True,
    strict: bool = True,
    num_questions: int = 20,
    tsv_file: os.PathLike = CIRCA,
    seed: int = 7,
) -> pd.DataFrame:

    NLI_MAPPING = {
        "entailment": "Yes",
        "neutral": "In the middle, neither yes nor no",
        "contradiction": "No",
    }
    np.random.seed(seed)

    df_circa = read_circa(tsv_file)
    if strict:
        goldstandard = "goldstandard1"
    else:
        goldstandard = "goldstandard2"

    df_circa = df_circa[
        (df_circa[goldstandard] != "nan") & (df_circa[goldstandard] != "Other")
    ]

    inputs = list(map(lambda x: eval(x).decode("utf-8"), open(inputs_file).readlines()))
    targets = list(map(eval, open(targets_file).readlines()))
    predictions = list(map(eval, open(predictions_file).readlines()))

    df = pd.DataFrame()
    df["inputs"] = inputs
    df["targets"] = targets
    df["predictions"] = predictions

    # process inputs
    df["inputs"] = df.inputs.apply(lambda x: x.split("hypothesis: ")[1].strip())
    df["canquestion-X"] = df.inputs.apply(
        lambda x: x.split("premise:")[0].strip()
    ).astype("string")
    df["answer-Y"] = df.inputs.apply(lambda x: x.split("premise:")[1].strip()).astype(
        "string"
    )
    df = (
        df.set_index(["canquestion-X", "answer-Y"])
        .join(df_circa.set_index(["canquestion-X", "answer-Y"]), how="inner")
        .reset_index()
    )

    # process targets
    df = df[df.targets.apply(lambda x: x["label"] != "none")]
    df["targets"] = df.targets.apply(lambda x: NLI_MAPPING[x["label"]])
    # TODO: check this after retraining
    # assert (df.targets != df.goldstandard1).all()

    # process predictions
    df["explanation"] = df.predictions.apply(lambda x: x["explanations"][0])
    df["predictions"] = df.predictions.apply(
        lambda x: NLI_MAPPING.get(x["label"], "None")
    )

    df = df[
        ["context", "question-X", "answer-Y", "targets", "predictions", "explanation"]
    ].sample(num_questions)
    df["concatenated"] = (
        "CONTEXT: "
        + df["context"]
        + "\t QUESTION(X): "
        + df["question-X"]
        + "\t ANSWER(Y): "
        + df["answer-Y"]
        + "\t TARGET: "
        + df["targets"]
        + "\t PREDICTION: "
        + df["predictions"]
        + "\t EXPLANATION:"
        + df["explanation"]
    )
    return df


if __name__ == "__main__":
    df = read_circa()
    # calc_bleu_scores()

    questions_interpret = gen_googleform_interpret()
    questions_interpret.to_csv("googleform_interpret.csv", sep="|")

    questions_explain = gen_googleform_explain_nli(
        "circa_inputs", "circa_targets", "circa_predictions"
    )
    questions_explain[['concatenated']].to_csv('googleform_explain.csv')
