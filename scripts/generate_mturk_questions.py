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


def gen_mturk_explain_nli(
    inputs_file: os.PathLike,
    targets_file: os.PathLike,
    predictions_file: os.PathLike,
    num_samples_correct: int = 20,
    num_samples_incorrect: int = 20,
    strict: bool = False,
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
    df["target"] = df.targets.apply(lambda x: NLI_MAPPING[x["label"]])

    # process predictions
    df["explanation"] = df.predictions.apply(lambda x: x["explanations"][0])
    df["prediction"] = df.predictions.apply(
        lambda x: NLI_MAPPING.get(x["label"], "None")
    )

    # select examples based on whether prediction is correct
    df = pd.concat(
        [
            df[df.prediction == df.target].sample(num_samples_correct),
            df[df.prediction != df.target].sample(num_samples_incorrect),
        ]
    )

    # coalesce target and prediction
    columns = ["context", "question-X", "answer-Y", "explanation"]
    df_target = df[columns + ["target"]].rename(columns={"target": "interpretation"})
    df_prediction = df[columns + ["prediction"]].rename(
        columns={"prediction": "interpretation"}
    )
    df = pd.concat([df_target, df_prediction]).drop_duplicates()

    return df.rename(columns={"question-X": "question", "answer-Y": "answer"})


if __name__ == "__main__":
    # df = read_circa()

    df = gen_mturk_explain_nli("circa_inputs", "circa_targets", "circa_predictions")
    df.to_csv("input_explanation.csv", index=False)
