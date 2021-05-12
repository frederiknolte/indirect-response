import csv
import math
import random
import os

from os import PathLike

def ensure_dir(file_path: PathLike) -> PathLike:
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

OUT_DIR = 'data'
IN_FILE = 'circa/circa-data.tsv'
SEEDS = [13, 948, 2756]
CIRCA_SIZE = 34268
SPLITS = [0.60, 0.20]

if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    train = math.floor(CIRCA_SIZE * SPLITS[0])
    test = math.floor(CIRCA_SIZE * SPLITS[1])
    validation = CIRCA_SIZE - train - test
    for seed in SEEDS:
        random.seed(seed)
        shuffled = list(range(CIRCA_SIZE))
        random.shuffle(shuffled)
        train_range, test_range, validation_range = shuffled[:train], shuffled[train:train+test], shuffled[-validation:]
        train_map = {i: "train" for i in train_range}
        test_map = {i: "test" for i in test_range}
        validation_map = {i: "val" for i in validation_range}
        assignment_map = {**train_map, **test_map, **validation_map}
        
        train_file = open(f"{OUT_DIR}/circa-{seed}-train.csv", "w+")
        test_file = open(f"{OUT_DIR}/circa-{seed}-test.csv", "w+")
        val_file = open(f"{OUT_DIR}/circa-{seed}-val.csv", "w+")

        with open(IN_FILE) as f:
            header = next(f)
            train_file.write(header)
            test_file.write(header)
            val_file.write(header)
            for i, line in enumerate(f):
                assignment = assignment_map[i]
                if assignment == "train":
                    train_file.write(line)
                elif assignment == "test":
                    test_file.write(line)
                elif assignment == "val":
                    val_file.write(line)
