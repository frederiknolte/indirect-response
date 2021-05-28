# Interpreting Indirect Answers Using Self-Rationalizing Models 🤔 ❓

This repository contains to run the experiments in our paper, `Interpreting Indirect Answers Using Self-Rationalizing Models`, where we finetune T5 [(Raffel et al., 2019)]((https://arxiv.org/abs/1910.10683)) to classify indirect answers and produce rationales which we evaluate with Leakage-Adjusted Simulatability [Hase et al., 2020](https://www.aclweb.org/anthology/2020.findings-emnlp.390.pdf) and human judgments of plausibility, faithfulness, and quality.

We wrote this paper for the Spring 2021 edition of the University of Amsterdam MSc Artificial Intelligence course, [Computational Dialogue Modelling](https://cl-illc.github.io/cdm/).

## 📖 Authors

1. [Yun Li](https://github.com/MotherOfUnicorns)
2. [Michael Neely](https://github.com/michaeljneely/)
3. [Frederik Nolte](https://github.com/frederiknolte)

## 👩‍🔬 Method

The Circa dataset [(Louis et al., 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.601/) of (question, indirect answer) pairs does not come with human-provided rationales. To teach our T5 model to rationalize (generate a natural language explanation) its predictions on the Circa dataset, we employ a mixture task with data from two datasets that do contain such rationales.

We finetune a [pretrained T5-Large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models) model on a mixture of data from the Circa, e-SNLI [(Camburu et al., 2018)](https://papers.nips.cc/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html) and Cos-E [(Rajani et al., 2019)](https://www.aclweb.org/anthology/P19-1487/) datasets. During the finetuning process we ask our T5 model to predict one of four labels in a Natural Language Inference style task. We also ask of our T5 model to rationalize some of its predictions on the e-SNLI and Cos-E datasets. Supervised rationale generation is possible for the latter two datasets, since they contain reference rationales.

We then evaluate our finetuned T5 by asking it to both predict and rationalize instances from a held out Circa test set. We measure predictive power with accuracy. We measure rationale quality with the Leakage-Adjusted Simulatibilty metric as well as with human annotations of quality with a Mechanical Turk survey.

### The Circa Dataset

We modify to Natural Language Inference setting to match the E-SNLI dataset  which supplies the bulk of the explanations from which the T5 model learns to rationalize. We use the `relaxed` Circa labels, mapped to the NLI setting as follows:

1. `Yes --> entailment`
2. `No --> contradiction`
3. `In the middle, neither yes nor no --> neutral`
4. `Yes, subject to some coniditions --> none`
5. `Other --> none`
6. `N/A --> none`

We begin all input sequences with `nli` keyword to tell T5 that this is an NLI task. If we want T5 to explain its prediction, we put the `explain` keyword before `nli`. We use the declarative form of the question (e.g., `Do you have any pets` becomes `You have pets`) as the **hypothesis**, prepended with a `hypothesis:` keyword. We use the answer (e.g., `My cat just turned one year old`) as the **premise** prepended with a `premise:` keyword.

The Circa Dataset has two settings:

1. **Matched**: where all response scenarios are seen during training.
2. **Unmatched**: where some response scenarios are held out in the validation and test sets.

In the **unmatched** setting, we add the response scenario to the input sequence prepended with a `context:` keyword.

Thus, we transform a relaxed Circa Example such as:

```code
    Scenario: Y has just travelled from a different city to meet X.
    Question: Did you drive straight here?
    Answer: I had to stop at my mom's house.
    Label: No
```

Into the following in the **matched setting** (with the optional explain keyword at the start):

```code
    input: nli hypothesis: I did drive straight here. premise: I had to stop at my mom's house.
    target: contradiction
```

And the following in the **unmatched setting**: (with the optional explain keyword at the start):

```code
    input: nli context: Y has just travelled from a different city to meet X. hypothesis: I did drive straight here. premise: I had to stop at my mom's house.
    target: contradiction
```

Since the Circa dataset has no pre-defined splits, we generate three unique splits based on three random seeds. Those are available in [circa/data/](./circa/data/)

### E-SNLI Dataset

We follow the method of [Narang et al., 2020](https://arxiv.org/pdf/2004.14546.pdf), turning e-SNLI instances to input sequences in a similar manner to the circa dataset. The only difference is that there is no context and no `none` label.

Thus, we tranformed an e-SNLI example such as:

```code
    premise: Cardinals lost last night.
    hypothesis: The Saint Louis Cardinals always win.
    label: contradiction
    explanation: you can't lose if you always win.
```

Into the following when we want T5 to only predict the correct label:

```code
    input: nli hypothesis: The Saint Louis Cardinals always win. premise: Cardinals lost last night.
    target: contradiction
```

And the following when we want T5 to both predict and rationalize:

```code
    input: explain nli hypothesis: The Saint Louis Cardinals always win. premise: Cardinals lost last night.
    target: contradiction. explanation: you can't lose if you always win.
```

### Cos-E Dataset

The Cos-E dataset provides common-sense explanations to some instances of the CommonSenseQA dataset [(Talmor et al., 2019)](https://www.aclweb.org/anthology/N19-1421/). Like Narang et al., 2020, we transform instances into an NLI/Question-Answering hybrid format, using a `premise` keyword for the question and `choice` keyword before each possible answer.

For example, we transform the following Cos-E instance:

```code
    question: Where can you go to use a piano in your neighborhood if you don't have one?
    choices: music school, music store, neighbor's house, lunch, drawing room
    correct answer: neighbor's house
    explanation: This word is the most relevant
```

Into the following when we want T5 to only predict the correct label:

```code
    input: nli premise: Where can you go to use a piano in your neighborhood if you don't have one? choice: music school choice: music store choice: neighbor's house choice: lunch choice: drawing room
    target: neighbor's house
```

And the following when we want T5 to both predict and rationalize:

```code
    input: explain nli premise: Where can you go to use a piano in your neighborhood if you don't have one? choice: music school choice: music store choice: neighbor's house choice: lunch choice: drawing room
    target: neighbor's house explanation: This word is the most relevant
```

## 🧪 Experiments

To reproduce our experiments on Google Cloud TPUs, please follow the instructions detailed in [finetune_on_gcp.md](finetune_on_gcp.md). We describe all of our experiments below. We conduct three runs of each experiment with three different seeds.

### Baselines

1. **Circa Baseline**: this involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them), thus setting the benchmark for accuracy.

2. **Circa Baseline - Hypothesis Only**: This involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them) without access to the hypothesis, thus setting the benchmark for accuracy in this ablated setting.

3. **Circa Baseline - Premise Only**: This involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them) without access to the premise, thus setting the benchmark for accuracy in this ablated setting.

4. **e-SNLI and Cos-E Zero Shot Transfer to Circa**: This involves finetuning the pretrained T5 model on a mixture of E-SNLI and Cos-E data and then performing zero shot evaluation (predictions with explanations) on the Circa Dataset.

### Mixture Experiments

In these experiments, we finetune the T5 model on a mixture of five different tasks. The mixing rate for each task is proportional to the size of its dataset:

1. Predicting e-SNLI instances

2. Predicting and rationalizing e-SNLI instances

3. Predicting Cos-E instances

4. Predicting and rationalizing Cos-E instances

5. Predicting Circa instances

We then select the model checkpoint with the highest accuracy on the Circa dataset and ask that model to both predict and rationalize all instances from the held out Circa test set. We train one model per seed in both the **matched** and **unmatched** setting.

## 💯 Evaluation

### Leakage-Adjusted Simulatibility

To train simulators and calculate LAS scores, please follow the instructions detailed in [run_LAS.md](running_LAS.md).

For each seed and each setting (**matched** and **unmatched**), we take the finetuned T5 model checkpoint with the highest Circa validation accuracy. We then generate rationales on the training and validation sets as well, so we have rationales for every instance from the Circa dataset.

We then use those predictions to train a DistilBERT simulator model per the method of Hase et al., 2020. Finally, we calculate the LAS scores.

### Human Survey

We use test set predictions from the unmatched model with the highest LAS score to generate our Mechanical Turk survey. @YUN: add more details and maybe a link to a README here

## Results

Read our paper, view our presentation, or explore the IPython notebooks in this repository to learn more!

## ❤️ Acknowledgments

We would like to thank Google for providing free access to Google Cloud `v3-8` TPUs via the [TPU Research Cloud](https://sites.research.google/trc/) program. We would also like to thank [Dr. Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/), [Mario Giulianelli](https://glnmario.github.io/) and [Ece Takmaz](https://ecekt.github.io/) for their assistance in conducting this research project.
