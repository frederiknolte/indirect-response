import csv
import functools
import itertools
import os
import re
import string

from collections import OrderedDict
from os import PathLike
from typing import List, Tuple, Union

import editdistance
import pandas as pd


# Common pronouns (including X and Y from the circa contexts)
PRONOUN_REGEX = r'(?:X|Y|I|I\'d|Id|I\'m|Im|you|we|he|she|they|I\'ve|Ive|you\'ve|youve|you\'re|youre|we\'ve|weve|we\'re|were|they\'ve|theyve|they\'re|theyre|a person|someone|the|it|one|the person|the people|my|its|it\'s)'

PRONOUN_LIST = ['X', 'Y', 'I', 'I\'d', 'Id', 'I\'m', 'Im', 'you', 'we', 'he', 'she', 'they', 'I\'ve', 'Ive', 'you\'ve', 'youve', 'you\'re', 'youre', 'we\'ve', 'weve', 'we\'re', 'were', 'they\'ve', 'theyve', 'they\'re', 'theyre', 'a person', 'someone', 'the', 'it', 'one', 'the person', 'the people', 'my', 'its', 'it\'s']

# We ignore case when pattern matching
def compile_regex(r_string: str) -> re.Pattern:
    return re.compile(r_string, flags=re.IGNORECASE)

# Ordered Dictionary because the least restrictive patterns are last
PATTERNS = OrderedDict({
    # ==== Entailment ==== #
    # If P then Q
    'implication': [
        compile_regex(r'^(?:if)\b(.*?)\b(?:then it is logical to conclude that| then it is logical to assume that|it is implied that|it follows that)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:implies|shows that)\b(.*?)$'),
    ],
    # P and Q are either equal or the set of P includes Q
    'equality': [
        compile_regex(r'^(.*?)\b(?:is the same as|is same as|is a form of|are a form of|is a type of|are a type of)\b(.*?)$'),
    ],
    # P and Q are semantically the same
    'rephrasing': [
        compile_regex(r'^(.*?)\b(?:is a rephrasing of|can also be said as|is a synonym of)\b(.*?)$'),
    ],
    # P is the only valid option for some statement Q
    'only_option': [
        compile_regex(r'^(.*?)\b(?:is the only|are the only)\b(.*?)$')
    ],
    # ==== Contradictions ==== #
    # The size of P is greater than the size of Q
    'comparison': [
        compile_regex(r'^(.*?)\b(?:is more than|are more than|is less than|are less than|is greater than|are greater than|are shorter than|are longer than)\b(.*?)$')
    ],
    # P and Q are opposites
    'opposite': [
        compile_regex(r'^(.*?)\b(?:is the opposite of|are opposites|are opposite)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:is different than|are different than|is different|are different)\b(.*?)$')
    ],
    # P and Q contradict each other
    'contradictory': [
        compile_regex(r'^(.*?)\b(?:is contradictory to)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:and)\b(.*?)\b(?:are contradictory pieces of information|are contradictory statements)\b(.*?)$')
    ],
    # P and Q cannot exist simultaneously
    'either or': [
        compile_regex(fr'^(?:{PRONOUN_REGEX} (?:is either|have either|cannot))\b(.*?)\b(?:or|and)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:cannot)\b(.*?)\b(?:at the same time.)$'),
        compile_regex(r'^(.*?)\b(?:cannot be both|cannot be)\b(.*?)$'),
    ],
    # ==== Neutral ==== #
    # The converse of a conditional statement is not always true
    'denying the consequent': [
        compile_regex(r'^(?:just because)\b(.*?)\b(?:does not mean|doesn\'t mean|does not necessarily mean|doesn\'t necessarily mean|doesn\'t mean that|does not mean that|does not indicate that|doesn\'t indicate that|doesn\'t indicate|does not indicate)\b(.*?)$'),
        compile_regex(r'^(?:the fact that|the fact)\b(.*?)\b(?:does not imply that|doesn\'t imply that)\b(.*?)$'),
        compile_regex(r'^\b(?:not all)\b(.*?)\b(?:are|is)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:is not always|is not necessarily|is not the only|does not imply|doesn\'t imply|does not imply that|doesn\'t imply that|does not mean|doesn\'t mean|does not necessarily mean|doesn\'t necessarily mean|doesn\'t mean that|does not mean that|does not indicate that|doesn\'t indicate that|doesn\'t indicate|does not indicate)\b(.*?)$'),
    ],
    # ==== Naive Logic (Very simple formulations of <some span> <some logical phrase> <some span>)==== #
    # P is not a Q
    'naive_negation': [
        compile_regex(r'^(.*?)\b(?:is not|are not|do not|does not|doesn\'t)\b(.*?)$'),
        compile_regex(fr'^(?:{PRONOUN_REGEX})\b(.*?)\b(?:so (?:{PRONOUN_REGEX}))\b(.*?)'),
        compile_regex(fr'^(?:{PRONOUN_REGEX})\b(.*?)\b(?:not)\b(.*?)'),
        compile_regex(fr'^(?:(?:if) {PRONOUN_REGEX})\b(.*?)\b(?:{PRONOUN_REGEX} (?:cant|can\'t|won\'t|wont))\b(.*?)'),
    ],
    # P is a Q
    'naive_assignment': [
        compile_regex(r'^(.*?)\b(?:is a|are a|are an|is an|is the|is|are|means)\b(.*?)$')
    ],
    # If P then Q
    'naive_conditional': [
        compile_regex(fr'^(?:{PRONOUN_REGEX})\b(.*?)\b(?:because)\b(.*?)'),
        compile_regex(fr'^(?:{PRONOUN_REGEX})\b(.*?)\b(?:so)\b(.*?)'),
        compile_regex(fr'^(?:(?:if) {PRONOUN_REGEX})\b(.*?)\b{PRONOUN_REGEX}\b(.*?)'),
        compile_regex(r'^(?:if)(.*?)\b(?:then|,then)\b(.*?)'),
        compile_regex(r'^(?:if)(.*?),(.*?)')
    ]
})


INPUTS = {
    'train': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_train_eval_circa_v100_nli_relaxed_unmatched13_inputs',
    'val': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_validation_eval_circa_eval_v100_nli_relaxed_unmatched13_inputs',
    'test': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_test_eval_circa_eval_v100_nli_relaxed_unmatched13_inputs'
}

PREDICTIONS = {
    'train': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_train_eval_circa_v100_nli_relaxed_unmatched13_1017500_predictions',
    'val': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_validation_eval_circa_eval_v100_nli_relaxed_unmatched13_1017500_predictions',
    'test': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_test_eval_circa_eval_v100_nli_relaxed_unmatched13_1017500_predictions'
}

TARGETS = {
    'train': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_train_eval_circa_v100_nli_relaxed_unmatched13_targets',
    'val': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_validation_eval_circa_eval_v100_nli_relaxed_unmatched13_targets',
    'test': 'predictions/esnli_and_cos_e_to_circa_nli_relaxed_unmatched13_test_eval_circa_eval_v100_nli_relaxed_unmatched13_targets'
}

LAS = {
    'train': 'predictions/LAS_nli_relaxed_unmatched13_test_data_circa_NLI_train.csv',
    'val': 'predictions/LAS_nli_relaxed_unmatched13_test_data_circa_NLI_dev.csv',
    'test': 'predictions/LAS_nli_relaxed_unmatched13_test_data_circa_NLI_test.csv'
}

# To make the regexes restrictive enough, we needed to add the pronouns in non-capture groups.
# So to check if the matches from the regexes align with the premise, hypothesis, context,
# and/or some combination of them, we need to add back in the pronouns
def is_same_but_missing_pronoun(
    candidates: Union[str, List[str]],
    reference: str
) -> bool:
    if isinstance(candidates, str):
        candidates = [candidates]
    for p in PRONOUN_LIST:
        with_added_pronouns = " ".join([f'{p} {c}' for c in candidates])
        if with_added_pronouns == reference:
            return True
    return False


# Account for minor mistakes when generating text, e.g., an extra comma or a missing letter
def within_edit_distance(
    candidate: str,
    reference: str,
    min_dist: int
) -> bool:
    if candidate is None or reference is None:
        return False
    return editdistance.eval(candidate, reference) <= min_dist


# Check if the candidate string is approximately close to some combination of the premise,
# hypothesis, and/or context.
def match_by_heuristic(
    candidate: str,
    context: str,
    premise: str,
    hypothesis: str,
    minimum_distance: int = 2
) -> Tuple[str, str]:
    within = functools.partial(within_edit_distance, min_dist=minimum_distance)

    if is_same_but_missing_pronoun(candidate, context) or within(candidate, context):
        return ('context', None)
    elif is_same_but_missing_pronoun(candidate, premise) or within(candidate, premise):
        return ('premise', None)
    elif is_same_but_missing_pronoun(candidate, hypothesis) or within(candidate, hypothesis):
        return ('hypothesis', None)
    spans = [
        ('context', context),
        ('premise', premise),
        ('hypothesis', hypothesis)
    ]
    for combo in itertools.product(spans, spans):
        tup1, tup2 = combo
        s1, val1 = tup1
        s2, val2 = tup2
        if within(candidate, val1 + val2) or is_same_but_missing_pronoun(candidate, [val1, val2]):
            return (s1, s2)
    return (None, None,)


def ensure_dir(file_path: PathLike) -> PathLike:
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


# Remove all punctuation and extra spaces
def remove_formatting(s: str) -> str:
    s = s.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    s = re.sub(' +', ' ', s)
    return s


# The labels from the LAS files are integers. Map back to the correct string.
def int_to_nli_label(x: int) -> str:
    mapping =  {
        0: 'neutral',
        1: 'entailment',
        2: 'contradiction',
        3: 'none'
    }
    return mapping[x]


def read_las_file(fp: PathLike) -> pd.DataFrame:
    return pd.read_csv(
        fp,
        index_col=0,
        sep=',',
        quoting=csv.QUOTE_NONE,
        encoding='utf-8',
        escapechar='\\'
    )


# Build one big frame for the LAS scores for each split in a method that can be joined with the T5 output frame
def build_combined_las_frame(frames: List[pd.DataFrame]) -> pd.DataFrame:
    cf = pd.concat(frames).reset_index()
    cf = cf.rename(columns={
        'explanation': 'rationale',
        'prediction': 'predicted_label',
        'target': 'target_label'
    })
    # Format data
    cf['target_label'] = cf['target_label'].apply(lambda x: int_to_nli_label(x))
    cf = cf[cf['target_label'] != 'none'] # drop instances without NLI label
    cf['predicted_label'] = cf['predicted_label'].apply(lambda x: int_to_nli_label(x))
    cf['premise'] = cf['premise'].astype(str).apply(lambda x: remove_formatting(x))
    cf['hypothesis'] = cf['hypothesis'].astype(str).apply(lambda x: remove_formatting(x))
    cf['context'] = cf['context'].astype(str).apply(lambda x: remove_formatting(x))
    cf['rationale'] = cf['rationale'].astype(str).apply(lambda x: remove_formatting(x))

    return cf


# Build a dataframe for the annotator scores in a method that can be joined with the T5 output frame
def build_annotator_frame(csv_paths: List[PathLike]) -> pd.DataFrame:
    af = pd.concat([
        pd.read_csv(fp) for fp in csv_paths
    ]).reset_index()
    # only keep necessary columns
    af = af[[
        'Input.context',
        'Input.answer',
        'Input.explanation',
        'Answer.explanation-quality.label',
    ]]
    af = af.rename(columns={
        'Input.context': 'context',
        'Input.answer': 'premise',
        'Input.explanation': 'rationale',
        'Answer.explanation-quality.label': 'quality',
    })
    # format data
    af['quality'] = af['quality'].apply(lambda x: int(x[0]))
    af['premise'] = af['premise'].astype(str).apply(lambda x: remove_formatting(x))
    af['context'] = af['context'].astype(str).apply(lambda x: remove_formatting(x))
    af['rationale'] = af['rationale'].astype(str).apply(lambda x: remove_formatting(x))
    # average across workers
    af = af.groupby(['premise', 'context', 'rationale'])['quality'].mean().reset_index()
    return af


# Build the T5 output frame from the inputs, targets, and predictions files for each split
def build_t5_outputs_frame() -> pd.DataFrame:

    df = {
        'template': [], # the name of the unique regex pattern
        'split': [], # train, test, or val
        'rationale': [], # the formatted rationale
        'span1_label': [], # either premise, hypothesis, context, or None
        'span2_label': [], # either premise, hypothesis, context, or None
        'predicted_label': [], # either entailment, contradiction, or neutral
        'target_label': [], # either entailment, contradiction, or neutral
        'premise': [], # the formatted premise
        'hypothesis': [], # the formatted hypothesis
        'context': [] # the formatted context
    }

    for split in ['train', 'val', 'test']:
        input_file, prediction_file, target_file = INPUTS[split], PREDICTIONS[split], TARGETS[split]
        input_handle, prediction_handle, target_handle = open(input_file), open(prediction_file), open(target_file)
        for raw_input, raw_prediction, raw_target in zip(input_handle, prediction_handle, target_handle):

            # read raw data
            raw_input, raw_prediction, raw_target = eval(raw_input), eval(raw_prediction), eval(raw_target)
            raw_input = raw_input.decode('utf-8')

            context, rest = raw_input.split('context: ')[1].split('hypothesis: ')
            hypothesis, premise = rest.split('premise: ')
            target = raw_target['label']
            prediction = raw_prediction['label']

            # drop instances without NLI label
            if raw_prediction['explanations']:
                rationale = raw_prediction['explanations'][0]
            else:
                continue

            # format data
            premise = remove_formatting(premise)
            hypothesis = remove_formatting(hypothesis)
            context = remove_formatting(context)

            rationale = remove_formatting(rationale)

            # match rationales by regex and name the pattern
            # also, check if the extracted capture groups exactly match the
            # premise, hypothesis, and/or context
            found_match = False
            pattern_name, span1_label, span2_label = None, None, None

            for name, patterns in PATTERNS.items():
                if not found_match:
                    for pattern in patterns:
                        match = re.match(pattern, rationale)
                        if match:
                            # anything past the first two matches is overflow
                            # but there should be at least two
                            assert len(match.groups()) >= 2
                            s1, s2 = match.group(1).strip(), match.group(2).strip()

                            s1 = remove_formatting(s1)
                            s2 = remove_formatting(s2)

                            span1_label = match_by_heuristic(s1, context, premise, hypothesis)
                            span2_label = match_by_heuristic(s2, context, premise, hypothesis)
                            pattern_name=name
                            found_match = True

                            break

            # Account for cases where the rationale is just a combination of the premise, hypothesis, and/or context
            if not found_match:
                span1_label, span2_label = match_by_heuristic(rationale, context, premise, hypothesis)
                pattern_name = span1_label or 'unique'
                if span2_label:
                    pattern_name = pattern_name + '_' + span2_label

                if pattern_name != 'unique':
                    pattern_name = 'parrot_' + pattern_name

            # add to frame
            df['split'].append(split)
            df['template'].append(pattern_name)
            df['predicted_label'].append(prediction)
            df['target_label'].append(target)
            df['span1_label'].append(span1_label)
            df['span2_label'].append(span2_label)
            df['premise'].append(premise)
            df['hypothesis'].append(hypothesis)
            df['context'].append(context)
            df['rationale'].append(rationale)

    return pd.DataFrame(df)


# Read, build, match, and write
if __name__ == "__main__":

    LAS_FRAME = build_combined_las_frame([
        read_las_file(LAS['train']),
        read_las_file(LAS['val']),
        read_las_file(LAS['test'])
    ])

    ANNOTATOR_FRAME = build_annotator_frame([
        'predictions/final_survey_results.csv',
    ])

    PATTERN_FRAME = build_t5_outputs_frame()

    COMBINED = pd.merge(
        left=PATTERN_FRAME,
        right=LAS_FRAME,
        how='inner',
        on=[
            'hypothesis',
            'premise',
            'context',
            'target_label',
            'predicted_label',
            'rationale'
        ]
    ).reset_index()


    COMBINED = pd.merge(
        left=COMBINED,
        right=ANNOTATOR_FRAME,
        how='left',
        on=[
            'premise',
            'context',
            'rationale'
        ]
    )

    COMBINED.to_pickle('artifacts/categorized_rationales.pkl')
