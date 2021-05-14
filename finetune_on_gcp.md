# To run experiments on Google Cloud

1. Open a cloud shell

2. Set some environment variables. If you're using the TPU Research Cloud credit, you probably won't have access to any beefier setup than `v3-8` TPUs. Note: **be sure to make a bucket first.**

    ```shell
    export PROJECT=your-google-cloud-project-name
    export ZONE=whatever-zone-you-get-free-tpus-in
    export BUCKET=gs://your-bucket
    export TPU=what-you-want-to-name-your-tpu
    export TOPOLOGY="v3-8"
    ```

3. Create a tpu cluster (from my experiments, n2-standard-2 has enough memory & storage but you might need more)

    ```shell
    gcloud compute tpus execution-groups create \
    --name="${TPU}" \
    --zone="${ZONE}" \
    --tf-version=2.4.1 \
    --machine-type=n2-standard-2 \
    --accelerator-type="${TOPOLOGY}"
    ```

4. It should automatically connect to your TPU. If not:

    ```shell
    gcloud compute ssh $TPU --zone $ZONE
    ```

5. Download the Indirect Response code

    ```shell
    git clone https://github.com/frederiknolte/indirect-response
    cd indirect_response
    git submodule init
    git submodule update
    ```

6. Download and install packages

    ```shell
    ./gcp_setup.sh
    source bin/activate
    cd google-research/wt5
    ```

7. Select your mixture task from those available [in this README](mixtures.md). Additionally, 

    ```shell
    export TASK=your-chosen-mixture-task
    export MODEL_DIR="${BUCKET}/${TASK}"
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

11. Pick the `.gin` sequence length file for whichever dataset in your mixture has the largest sequences. This file controls the padding. You wouldn't want your sequences getting cut off! If you're using the `esnli` dataset, it's probably the longest.

    ```shell
    export SEQ_LENGTH_FILE="wt5/gin/sequence_lengths/cos_e_v001.gin"
    ```

12. Execute your mixture task to finetune the model. Ensure the parent directory (`google-research`) is included in the `PYTHONPATH`:

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
    --gin_param="utils.run.skip_seen_data = True" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa_splits.circa_matched${RANDOM_SEED}" \
    --gin_location_prefix="wt5/wt5/gin/"
    ```

13. Run your evaluation task

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
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="mesh_eval_dataset_fn.use_cached=False" \
    --gin_param="utils.run.dataset_split = 'validation'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="utils.run.eval_checkpoint_step='all'" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa_splits.circa_matched${RANDOM_SEED}" \
    --gin_location_prefix="wt5/wt5/gin/" \
    --gin_param="utils.run.eval_summary_dir='${MODEL_DIR}/validation_eval'"
    ```

14. Run the final evaluation on the Circa test set

    ```shell
    TBD
    ```
