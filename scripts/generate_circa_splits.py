"""
Generate deterministic splits of the matched and unmatched settings of the Circa dataset for each random seed supplied
"""
import argparse
import csv
import math
import random
import os

def ensure_dir(file_path: os.PathLike) -> os.PathLike:
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

# Map each unique Circa context to an integer ID
CONTEXT_ID_MAP = {
    "X wants to know about Y's food preferences.": 0,
    "X wants to know what activities Y likes to do during weekends.": 1,
    "X wants to know what sorts of books Y likes to read.": 2,
    "Y has just moved into a neighbourhood and meets his/her new neighbour X.": 3,
    "X and Y are colleagues who are leaving work on a Friday at the same time.": 4,
    "X wants to know about Y's music preferences.": 5,
    "Y has just travelled from a different city to meet X.": 6,
    "X and Y are childhood neighbours who unexpectedly run into each other at a cafe.": 7,
    "Y has just told X that he/she is thinking of buying a flat in New York.": 8,
    "Y has just told X that he/she is considering switching his/her job.": 9
}

# Static values of dataset
NUM_CONTEXTS = len(CONTEXT_ID_MAP)
CIRCA_SIZE = 34268

## Defaults for user controlled params
OUT_DIR = 'data'
IN_FILE = 'circa/circa-data.tsv'
SEEDS = [13, 948, 2756]
SPLITS = [0.60, 0.20]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate splits of the Circa dataset')

    parser.add_argument(
        "--in-file",
        metavar="in_file",
        help="Path to the 'circa-data.tsv' file",
        default=IN_FILE
    )

    parser.add_argument(
        "--out-dir",
        metavar="out_dir",
        help="Directory where the splits will be written. Default='data'",
        default=OUT_DIR
    )

    parser.add_argument(
        "--splits",
        nargs=2,
        metavar=('train_split', 'test_split'),
        help="Percentage of data (e.g., 0.6, 0.2...) assigned to train and test folds. The remaining data is assigned to the validation fold.",
        type=float,
        default=SPLITS
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        help="Random seeds. One random split of the matched and unmatched settings will be created per seed",
        type=int,
        default=SEEDS
    )

    args = parser.parse_args()

    assert sum(args.splits) <= 1
    TRAIN_SPLIT, TEST_SPLIT = args.splits

    ensure_dir(args.out_dir)

    # Calculate the size of each fold. The validation set gets any leftovers from rounding.
    # Matched setting. Size based on number of instances.
    train_size = math.floor(CIRCA_SIZE * TRAIN_SPLIT)
    test_size = math.floor(CIRCA_SIZE * TEST_SPLIT)
    val_size = CIRCA_SIZE - train_size - test_size

    # Unmatched setting. Size based on number of contexes.
    train_contexes = math.floor(NUM_CONTEXTS * TRAIN_SPLIT)
    test_contexes = math.floor(NUM_CONTEXTS * TEST_SPLIT)
    val_contexes = NUM_CONTEXTS - train_contexes - test_contexes

    # Generate one deterministic dataset split per random seed supplied
    for seed in args.seeds:

        # Shuffle Context and Instance IDs and split into folds
        random.seed(seed)
        indices = list(range(CIRCA_SIZE))
        contexes = list(range(NUM_CONTEXTS))
        random.shuffle(indices)
        random.shuffle(contexes)
        train_range_idx = indices[:train_size]
        train_range_context = contexes[:train_contexes]
        test_range_idx = indices[train_size: train_size + test_size]
        test_range_context = contexes[train_contexes: train_contexes + test_contexes]
        val_range_idx = indices[-val_size:]
        val_range_context = contexes[-val_contexes:]

        # Build maps that assign each instance_id and context to a particular fold
        train_idx_map = {i: "train" for i in train_range_idx}
        test_idx_map = {i: "test" for i in test_range_idx}
        val_idx_map = {i: "val" for i in val_range_idx}
        assignment_idx_map = {**train_idx_map, **test_idx_map, **val_idx_map}

        train_context_map = {i: "train" for i in train_range_context}
        test_context_map = {i: "test" for i in test_range_context}
        val_context_map = {i: "val" for i in val_range_context}
        assignment_context_map = {**train_context_map, **test_context_map, **val_context_map}

        # Parallel file writing
        train_file_matched = open(f"{args.out_dir}/circa-train-matched-{seed}.tsv", "w+")
        test_file_matched = open(f"{args.out_dir}/circa-test-matched-{seed}.tsv", "w+")
        val_file_matched = open(f"{args.out_dir}/circa-val-matched-{seed}.tsv", "w+")

        train_file_unmatched = open(f"{args.out_dir}/circa-train-unmatched-{seed}.tsv", "w+")
        test_file_unmatched = open(f"{args.out_dir}/circa-test-unmatched-{seed}.tsv", "w+")
        val_file_unmatched = open(f"{args.out_dir}/circa-val-unmatched-{seed}.tsv", "w+")


        with open(args.in_file) as f:
            # Write tsv headers
            header = next(f)
            train_file_matched.write(header)
            test_file_matched.write(header)
            val_file_matched.write(header)
            train_file_unmatched.write(header)
            test_file_unmatched.write(header)
            val_file_unmatched.write(header)

            for line in f:

                # Matched: asssign instance by its ID
                instance_id = int(line.split('\t')[0])

                idx_assignment = assignment_idx_map[instance_id]
                if idx_assignment == "train":
                    train_file_matched.write(line)
                elif idx_assignment == "test":
                    test_file_matched.write(line)
                elif idx_assignment == "val":
                    val_file_matched.write(line)

                # Unmatched: assign instance by context ID
                context = line.split('\t')[1]
                context_id = CONTEXT_ID_MAP[context]
                context_assignment = assignment_context_map[context_id]

                if context_assignment == "train":
                    train_file_unmatched.write(line)
                elif context_assignment == "test":
                    test_file_unmatched.write(line)
                elif context_assignment == "val":
                    val_file_unmatched.write(line)

