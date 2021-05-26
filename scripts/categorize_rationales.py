import re
from collections import OrderedDict
import pandas as pd
import string
import editdistance
import os
from os import PathLike

def compile_regex(r_string):
    return re.compile(r_string, flags=re.IGNORECASE)

PATTERNS = OrderedDict({
    # ==== Entailment ==== #
    'conditional_statement': [
        compile_regex(r'^(?:if)\b(.*?)\b(?:then it is logical to conclude that| then it is logical to assume that|it is implied that|it follows that)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:implies|shows that)\b(.*?)$'),
    ],
    'equality': [
        compile_regex(r'^(.*?)\b(?:is the same as|is same as|is a form of|are a form of|is a type of|are a type of)\b(.*?)$'),
    ],
    'rephrasing': [
        compile_regex(r'^(.*?)\b(?:is a rephrasing of|can also be said as|is a synonym of)\b(.*?)$'),
    ],
    'only_option': [
        compile_regex(r'^(.*?)\b(?:is the only|are the only)\b(.*?)$')
    ],
    # ==== Contradictions ==== # 
    'comparison': [
        compile_regex(r'^(.*?)\b(?:is more than|are more than|is less than|are less than|is greater than|are greater than|are shorter than|are longer than)\b(.*?)$')
    ],
    'opposite': [
        compile_regex(r'^(.*?)\b(?:is the opposite of|are opposites|are opposite)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:is different than|are different than|is different|are different)\b(.*?)$')
    ],
    'contradictory': [
        compile_regex(r'^(.*?)\b(?:is contradictory to)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:and)\b(.*?)\b(?:are contradictory pieces of information|are contradictory statements)\b(.*?)$')
    ],
    'either or': [
        compile_regex(r'^(?:|I have either )(.*?)\b(?:or)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:is either )(.*?)\b(?:or)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:cannot)\b(.*?)\b(?:at the same time.)$'),
        compile_regex(r'^(.*?)\b(?:cannot be both|cannot be)\b(.*?)$')
    ],
    # ==== Neutral ==== #
    # The converse of a conditional statement is not always true
    'denying the consequent': [
        compile_regex(r'^(?:just because)\b(.*?)\b(?:does not mean|doesn\'t mean|does not necessarily mean|doesn\'t necessarily mean|doesn\'t mean that|does not mean that|does not indicate that|doesn\'t indicate that|doesn\'t indicate|does not indicate)\b(.*?)$'),
        compile_regex(r'^(?:the fact that|the fact)\b(.*?)\b(?:does not imply that|doesn\'t imply that)\b(.*?)$'),
        compile_regex(r'^\b(?:not all)\b(.*?)\b(?:are|is)\b(.*?)$'),
        compile_regex(r'^(.*?)\b(?:is not always|is not necessarily|is not the only|does not imply|doesn\'t imply|does not imply that|doesn\'t imply that|does not mean|doesn\'t mean|does not necessarily mean|doesn\'t necessarily mean|doesn\'t mean that|does not mean that|does not indicate that|doesn\'t indicate that|doesn\'t indicate|does not indicate)\b(.*?)$'),
    ],
    # ==== Naive (Very simple formulations of <some span> <some logical phrase> <some span>)==== #
    'naive_negation': [
        compile_regex(r'^(.*?)\b(?:is not|are not|do not|does not|doesn\'t)\b(.*?)$')
    ],
    'naive_assignment': [
        compile_regex(r'^(.*?)\b(?:is a|are a|is an|is a type of|are a type of|is the|is|are|means)\b(.*?)$')
    ],
    'naive_conditional': [
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

def match_by_heuristic(
    candidate: str,
    context: str,
    premise: str,
    hypothesis: str,
    minimum_distance: int = 2
) -> str:
    if editdistance.eval(candidate, context) <= minimum_distance:
        return "context"
    elif editdistance.eval(candidate, premise) <= minimum_distance:
        return "premise"
    elif editdistance.eval(candidate, hypothesis) <= minimum_distance:
        return "hypothesis"
    else:
        return None

def ensure_dir(file_path: PathLike) -> PathLike:
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

if __name__ == "__main__":

    df = {
        'pattern': [],
        'split': [],
        'rationale': [],
        'span_1': [],
        'span_2': [],
        'predicted_label': [],
        'target_label': []
    }

    good_matches = 0
    naive_matches = 0
    total_instances = 0

    for split in list(INPUTS.keys()):
        input_file, prediction_file, target_file = INPUTS[split], PREDICTIONS[split], TARGETS[split]
        input_handle, prediction_handle, target_handle = open(input_file), open(prediction_file), open(target_file)
        for raw_input, raw_prediction, raw_target in zip(input_handle, prediction_handle, target_handle):

            raw_input, raw_prediction, raw_target = eval(raw_input), eval(raw_prediction), eval(raw_target)
            raw_input = raw_input.decode('utf-8')

            context, rest = raw_input.split('context: ')[1].split('hypothesis: ')
            hypothesis, premise = rest.split('premise: ')
            target = raw_target['label']
            prediction = raw_prediction['label']

            premise= premise.translate(str.maketrans('', '', string.punctuation)).strip().lower()
            hypothesis = hypothesis.translate(str.maketrans('', '', string.punctuation)).strip().lower()
            context = context.translate(str.maketrans('', '', string.punctuation)).strip().lower()

            if raw_prediction['explanations']:
                rationale = raw_prediction['explanations'][0]
                total_instances += 1
            else:
                continue

            found_match = False

            NAME, SPAN1, SPAN2 = None, None, None

            for name, patterns in PATTERNS.items():
                if not found_match:
                    for pattern in patterns:
                        match = re.match(pattern, rationale)
                        if match:
                            assert len(match.groups()) >= 2 # anything past the first two matches is overflow
                            s1, s2 = match.group(1).strip(), match.group(2).strip()

                            s1 = s1.translate(str.maketrans('', '', string.punctuation)).strip().lower()
                            s2 = s2.translate(str.maketrans('', '', string.punctuation)).strip().lower()

                            c2n = {
                                premise: 'premise',
                                hypothesis: 'hypothesis',
                                context: 'context'
                            }

                            SPAN1 = c2n.get(s1) or match_by_heuristic(s1, context, premise, hypothesis)
                            SPAN2 = c2n.get(s2) or match_by_heuristic(s2, context, premise, hypothesis)
                            NAME=name
                            found_match = True
                            
                            if 'naive' in name:
                                naive_matches += 1
                            else:
                                good_matches += 1

                            break

            # Account for the odd case where the model ju
            if not found_match:
                SPAN_1 = match_by_heuristic(rationale, context, premise, hypothesis)
                if SPAN1:
                    NAME = 'direct_match'
                    naive_matches += 1
                else:
                    NAME = 'unique'

            df['split'].append(split)
            df['pattern'].append(NAME)
            df['predicted_label'].append(prediction)
            df['target_label'].append(target)
            df['span_1'].append(SPAN1)
            df['span_2'].append(SPAN2)
            df['rationale'].append(rationale)

    good_accounted_for = round(good_matches / total_instances, 2) * 100
    naive_accounted_for = round(naive_matches / total_instances, 2) * 100
    num_patterns = len(list(PATTERNS.keys())) - 3


    print(f'{num_patterns} unique patterns account for {good_accounted_for}%of rationales')
    print(f'3 naive patterns account for {naive_accounted_for}%')
    print(f'Explore the DataFrame located at \'artifacts/rationale_patterns.pkl\' for more information')

    ensure_dir('artifacts/')
    pd.DataFrame(df).to_pickle('artifacts/rationale_patterns.pkl')
