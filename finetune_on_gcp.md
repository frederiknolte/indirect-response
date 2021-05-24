# To run experiments on Google Cloud

## Setup

1. Open a cloud shell

2. Create a tpu cluster (from our experiments, n2-standard-2 has enough memory & storage but you might need more). If you're using the TPU Research Cloud credit, you probably won't have access to any beefier setup than `v3-8` TPUs.

    ```shell
    gcloud compute tpus execution-groups create \
    --name=what-you-want-to-name-your-tpu \
    --zone=whatever-zone-you-get-free-tpus-in \
    --tf-version=2.4.1 \
    --machine-type=n2-standard-2 \
    --accelerator-type="v3-8"
    ```

3. It should automatically connect to your TPU. If not:

    ```shell
    gcloud compute ssh whatever-you-named-your-tpu --zone whatever-zone-you-chose
    ```

4. Start a `tmux` session to prevent your commands from being interrupted from disconnects

    ```shell
        tmux
    ```

5. Set some environment variables. Note: **be sure to make a bucket first.**

    ```shell
    export PROJECT=your-google-cloud-project-name
    export ZONE=whatever-zone-you-chose
    export BUCKET=gs://your-bucket
    export TPU=whatever-you-named-your-tpu
    export TOPOLOGY=v3-8
    ```

6. Download the Indirect Response code

    ```shell
    git clone https://github.com/frederiknolte/indirect-response
    cd indirect_response
    git submodule init
    git submodule update
    ```

7. Download and install packages

    ```shell
    ./gcp_setup.sh
    source bin/activate
    cd google-research/wt5
    ```

8. Pick one of the [T5 models pretrained by Google](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models). Note the latest checkpoint of that model. That is the value you need to provide for `PRETRAINED_STEPS`.

    ```shell
    export PRETRAINED_DIR=gs://t5-data/pretrained_models/large
    export PRETRAINED_STEPS=1000700
    ```

9. Select the number of steps you would like to finetune for

    ```shell
    export FINETUNE_STEPS=20000
    ```

10. Pick one of the random seeds used to split the Circa dataset. Currently, those are: `13`, `948`, or `2756`.

    ```shell
    export RANDOM_SEED=13
    ```

11. Select the Circa dataset type you would like to use (`matched` or `unmatched`):

    ```shell
    export CIRCA_TYPE=matched
    ```

12. Pick the `.gin` sequence length file for whichever dataset in your mixture has the largest sequences. This file controls the maximum sequence sequence. By default we use use `{'inputs': 512, 'targets': 256}` for everything.

    ```shell
    export SEQ_LENGTH_FILE=wt5/gin/sequence_lengths/esnli_v002.gin
    ```

## Running Experiments

The following commands hold for all experiments. You just need to set the appropriate environment variables first. All available mixtures are listed and explained [in this README](mixtures.md).

### Training

```shell
PYTHONPATH=$PYTHONPATH:/home/$USER/indirect-response/google-research/ t5_mesh_transformer \
    --tpu="${TPU}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
    --gin_file="${SEQ_LENGTH_FILE}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TOPOLOGY}'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="mesh_train_dataset_fn.use_cached=False" \
    --gin_param="utils.run.save_checkpoints_steps=1000" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
    --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
    --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.constant_learning_rate" \
    --gin_param="constant_learning_rate.learning_rate=1e-3" \
    --gin_param="mesh_train_dataset_fn.seed=${RANDOM_SEED}" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa_splits.circa_${CIRCA_TYPE}${RANDOM_SEED}" \
    --gin_location_prefix="wt5/wt5/gin/"
```

### Validation

To validate the model you need to set the `VAL_TASK` variable. The value depends on what you want to validate. For example, if you want to validate every task in your training mixture, you can set `VAL_TASK` to the same value as `TASK`.

```shell
PYTHONPATH=$PYTHONPATH:/home/$USER/indirect-response/google-research/ t5_mesh_transformer \
    --tpu="${TPU}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="${SEQ_LENGTH_FILE}" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TOPOLOGY}'" \
    --gin_param="MIXTURE_NAME = '${VAL_TASK}'" \
    --gin_param="mesh_eval_dataset_fn.use_cached=False" \
    --gin_param="utils.run.dataset_split = 'validation'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --gin_param="mesh_eval_dataset_fn.seed=${RANDOM_SEED}" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa_splits.circa_${CIRCA_TYPE}${RANDOM_SEED}" \
    --gin_location_prefix="wt5/wt5/gin/" \
    --gin_param="utils.run.eval_summary_dir='${MODEL_DIR}/validation_eval'"
```

### Test Set Evaluation

Once you've run the validation task, it's up to you to find the best checkpoint for the final test set evaluation. Set that variable first:

```shell
    export BEST_VAL_CHECKPOINT=...
```

Then set the name of the task/mixture you would like to use:

```shell
    export FINAL_EVAL_TASK=...
```

Finally, run the evaluation:

```shell
    PYTHONPATH=$PYTHONPATH:/home/$USER/indirect-response/google-research/ t5_mesh_transformer \
    --tpu="${TPU}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --gin_file="${SEQ_LENGTH_FILE}" \
    --gin_file="eval.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TOPOLOGY}'" \
    --gin_param="MIXTURE_NAME = '${FINAL_EVAL_TASK}'" \
    --gin_param="mesh_eval_dataset_fn.use_cached=False" \
    --gin_param="utils.run.dataset_split = 'test'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="utils.run.eval_checkpoint_step=${BEST_VAL_CHECKPOINT}" \
    --gin_param="mesh_eval_dataset_fn.seed=${RANDOM_SEED}" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa_splits.circa_${CIRCA_TYPE}${RANDOM_SEED}" \
    --gin_location_prefix="wt5/wt5/gin/" \
    --gin_param="utils.run.eval_summary_dir='${MODEL_DIR}/test_eval'"
```

## Running the Baselines

Here we list the variables we set to run the baseline experiments. Refer to the previous section to see how to train, validate, and evaluate on the appropriate test set.

### Circa Baselines

This involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them), thus setting the benchmark for accuracy.

1. Variables necessary to train:

    ```shell
        export TASK="circa_v100_0_expln_nli_relaxed_${CIRCA_TYPE}${RANDOM_SEED}"
        export MODEL_DIR="${BUCKET}/${TASK}"
    ```

2. Variables necessary to validate:

    ```shell
        export EVAL_TASK="${TASK}"
    ```

3. Then set your best checkpoint and evaluate on the test set:

    ```shell
        export BEST_VAL_CHECKPOINT=...
        export FINAL_EVAL_TASK="${TASK}"
    ```

### Circa Baselines - Premise Only

This involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them) without access to the hypothesis, thus setting the benchmark for accuracy in this ablated setting.

1. Variables necessary to train:

    ```shell
        export TASK="circa_nli_baseline_premise_only_relaxed_${CIRCA_TYPE}${RANDOM_SEED}"
        export MODEL_DIR="${BUCKET}/${TASK}"
    ```

2. Variables necessary to validate:

    ```shell
        export EVAL_TASK="${TASK}"
    ```

3. Then set your best checkpoint and evaluate on the test set:

    ```shell
        export BEST_VAL_CHECKPOINT=...
        export FINAL_EVAL_TASK="${TASK}"
    ```

### Circa Baselines - Hypothesis Only

This involves finetuning the pretrained T5 model to generate the correct NLI labels for the Circa dataset (without explaining them) without access to the premise, thus setting the benchmark for accuracy in this ablated setting.

1. Variables necessary to train:

    ```shell
        export TASK="circa_nli_baseline_hypothesis_only_relaxed_${CIRCA_TYPE}${RANDOM_SEED}"
        export MODEL_DIR="${BUCKET}/${TASK}"
    ```

2. Variables necessary to validate:

    ```shell
        export EVAL_TASK="${TASK}"
    ```

3. Then set your best checkpoint and evaluate on the test set:

    ```shell
        export BEST_VAL_CHECKPOINT=...
        export FINAL_EVAL_TASK="${TASK}"
    ```

### Zero Shot Transfer to Circa from E-SNLI and Cos-E

This involves finetuning the pretrained T5 model on a mixture of E-SNLI and Cos-E data and then performing zero shot evaluation (predictions with explanations) on the Circa Dataset.

1. Variables necessary to train:

    ```shell
        export TASK="esnli_and_cos_e_to_circa_zero_shot"
        export MODEL_DIR="${BUCKET}/${TASK}"
    ```

2. Variables necessary to validate:

    ```shell
        export EVAL_TASK="circa_eval_v100_nli_relaxed_${CIRCA_TYPE}${RANDOM_SEED}"
    ```

3. Then set your best checkpoint and evaluate on the test set:

    ```shell
        export BEST_VAL_CHECKPOINT=...
        export FINAL_EVAL_TASK="${EVAL_TASK}"
    ```

## Running the Finetuning Experiments

This involves finetuning the pretrained T5 model on a mixture of E-SNLI and Cos-E data and Circa data (without generating explanations) and then evaluating the model's capability to generate predictions **and** explanations for the Circa dataset.

1. Variables necessary to train:

    ```shell
        export TASK="esnli_and_cos_e_to_circa_nli_${CIRCA_TYPE}${RANDOM_SEED}"
        export MODEL_DIR="${BUCKET}/${TASK}"
    ```

2. Variables necessary to validate:

    ```shell
        export EVAL_TASK="circa_eval_v100_nli_relaxed_${CIRCA_TYPE}${RANDOM_SEED}"
    ```

3. Then set your best checkpoint and evaluate on the test set:

    ```shell
        export BEST_VAL_CHECKPOINT=...
        export FINAL_EVAL_TASK="${EVAL_TASK}"
