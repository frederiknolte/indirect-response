import csv
import os

import numpy as np
import pandas as pd
import sacrebleu

CIRCA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../circa/circa-data.tsv"
)

MTURK_SCREENING = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "input_interpretation.csv"
)


def read_mturk_screening_questions(mturk_input=MTURK_SCREENING):
    """make sure we do not show screening questions to the annotators again"""
    df_screening = pd.read_csv(mturk_input)

    df = pd.DataFrame(columns=["context", "question-X", "answer-Y"])
    for ct in (1, 2, 3):
        df = df.append(
            {
                "context": df_screening.loc[0, f"context{ct}"],
                "question-X": df_screening.loc[0, f"question{ct}"],
                "answer-Y": df_screening.loc[0, f"answer{ct}"],
            },
            ignore_index=True,
        )

    # also append all examples we used for instructions
    examples = [
        [
            "X wants to know about Y’s food preferences.",
            "Do you eat red meat?",
            "I am a vegetarian.",
        ],
        [
            "X wants to know about Y's movie preferences.",
            "Do you like movies with sad endings?",
            "I often watch them.",
        ],
        [
            "X wants to know about Y's movie preferences.",
            "Are you up for a movie?",
            "Only to a comedy.",
        ],
        [
            "Y has just told X that he/she is considering switching his/her job.",
            "Will you have along commute?",
            "I'll be living very close to the job.",
        ],
        [
            "Y has just told X that he/she is considering switching his/her job.",
            "Are you excited to start a new job?",
            "I have mixed feelings.",
        ],
        [
            "Y has just told X that he/she is thinking of buying a flat in New York.",
            "Is a basement flat okay?",
            "I need to be above ground. ",
        ],
        [
            "X wants to know about Y's food preferences.",
            "Are you allergic to seafood?",
            "Avoiding seafood is best for my health.",
        ],
        [
            "X and Y are colleagues who are leaving work on a Friday at the same time.",
            "Got any plans?",
            "My schedule is open.",
        ],
        [
            "X wants to know what activities Y likes to do during weekends.",
            "Are you a fan of bars?",
            "I'm in AA.",
        ],
        [
            "Y has just told X that he/she is considering switching his/her job.",
            "Are you in a support position?",
            "I'm a supervisor.",
        ],
        [
            "X wants to know what activities Y likes to do during weekends.",
            "Is disk golf fun to you?",
            "I've never done it.",
        ],
        [
            "Y has just moved into a neighbourhood and meets his/her new neighbour X.",
            "Are you extroverted?",
            "I don't know",
        ],
        [
            "X and Y are childhood neighbours who unexpectedly run into each other at a cafe.",
            "Do you still talk to our friend Katie?",
            "Remind me who she is",
        ],
    ]

    for c, q, a in examples:
        df = df.append(
            {"context": c, "question-X": q, "answer-Y": a}, ignore_index=True
        )

    return df


def read_circa_original(
    tsv_file: os.PathLike = CIRCA,
    relaxed: bool = True,
) -> pd.DataFrame:

    df = pd.read_csv(CIRCA, delimiter="\t")
    for col in df.columns:
        if col == "id":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype("string")

    if relaxed:
        goldstandard = "goldstandard2"
    else:
        goldstandard = "goldstandard1"
    df = df[(df[goldstandard] != "nan") & (df[goldstandard] != "Other")]

    # remove conversations used as examples
    df_examples = read_mturk_screening_questions()
    df_examples["is_example"] = True
    df = df.merge(df_examples, on=["context", "question-X", "answer-Y"], how="left")

    df = df[df.is_example.isna()]
    df.drop(columns=["is_example"], inplace=True)

    return df


def gen_mturk_explain_nli_relaxed(
    las_file: os.PathLike,
    num_samples_per_category: int = 10,
    tsv_file: os.PathLike = CIRCA,
    seed: int = 7,
    exclude_samples=None,
) -> pd.DataFrame:
    """sample explanations from circa+LAS test data, from 4 categories:
    - prediction correct/incorrect
    - LAS leaked/nonleaked
    (for RELAXED setting)
    """
    np.random.seed(seed)

    df_circa = read_circa_original(tsv_file)
    goldstandard = "goldstandard2"
    df_circa = df_circa[
        (df_circa[goldstandard] != "nan") & (df_circa[goldstandard] != "Other")
    ]

    NLI_MAP = {
        "entailment": "Yes",
        "neutral": "In the middle, neither yes nor no",
        "contradiction": "No",
    }
    LABEL_MAP = {0: "neutral", 1: "entailment", 2: "contradiction", 3: "none"}

    df = pd.read_csv(las_file, sep=",", quoting=csv.QUOTE_NONE, escapechar="\\")
    df = df[["hypothesis", "premise", "target", "prediction", "explanation", "leaked"]]
    # a hack: the unmatched case has some parts of the context in "hypothesis" as well
    df["hypothesis"] = df["hypothesis"].apply(
        lambda x: str(x).split(" hypothesis: ")[-1]
    )
    df["target"] = df.target.apply(lambda x: NLI_MAP.get(LABEL_MAP[x], None))
    df["prediction"] = df.prediction.apply(lambda x: NLI_MAP.get(LABEL_MAP[x], None))
    df = df[~(df.target.isna() | df.prediction.isna())]

    df = (
        df.set_index(["hypothesis", "premise", "target"])
        .join(
            df_circa.rename(
                columns={
                    "canquestion-X": "hypothesis",
                    "answer-Y": "premise",
                    "goldstandard2": "target",
                }
            ).set_index(["hypothesis", "premise", "target"]),
            how="inner",
        )
        .reset_index()
    )
    # there are a few converstaions that have the same Q-A-goldstandard2 but different judgements
    # since we only care about the RELAXED setting we can ignore them
    df.drop_duplicates(
        ["hypothesis", "premise", "target", "prediction", "explanation", "leaked"],
        inplace=True,
    )

    if exclude_samples:
        df_exclude = pd.read_csv(exclude_samples).rename(
            columns={"question": "question-X", "answer": "premise"}
        )
        df_exclude["exclude"] = True
        df = df.merge(df_exclude, on=["context", "question-X", "premise", 'explanation'], how="left")
        df = df[df.exclude.isna()]

    df["correct_pred"] = df.target == df.prediction
    print(df.groupby(["correct_pred", "leaked"]).apply(lambda _df: _df.shape[0]))
    df = (
        df.groupby(["correct_pred", "leaked"])
        .apply(lambda _df: _df.sample(num_samples_per_category))
        .reset_index(drop=True)
    ).rename(columns={"question-X": "question", "premise": "answer"})

    # coalesce target and prediction
    columns = ["context", "question", "answer", "explanation"]
    df_target = df[columns + ["target"]].rename(columns={"target": "interpretation"})
    df_prediction = df[columns + ["prediction"]].rename(
        columns={"prediction": "interpretation"}
    )
    df_mturk = pd.concat([df_target, df_prediction]).drop_duplicates()

    return df, df_mturk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="input csv file from LAS")
    parser.add_argument("--output_csv", help="output csv file for MTurk")
    parser.add_argument("--num_sample", type=int, help="number of samples per category")
    parser.add_argument(
        "--exclude_samples", help="csv file containing samples to exclude"
    )
    args = parser.parse_args()

    # df, df_mturk = gen_mturk_explain_nli_relaxed("matched_data/circa/NLI/test.csv")
    # df.to_csv("input_explanation_original.csv", index=False)
    # df_mturk.to_csv("input_explanation.csv", index=False)
    df, df_mturk = gen_mturk_explain_nli_relaxed(
        args.input_csv,
        num_samples_per_category=args.num_sample,
        exclude_samples=args.exclude_samples,
    )
    df.to_csv("original" + args.output_csv, index=False)
    df_mturk.to_csv(args.output_csv, index=False)
