# To run experiments on Google Cloud

1. Open a cloud shell

2. Set some environment variables. Be sure to make a bucket first.

    ```shell
    export PROJECT=your-google-cloud-project-name
    export ZONE=whatever-zone-you-get-free-tpus-in
    export BUCKET=gs://your-bucket
    export TPU=what-you-want-to-name-your-tpu
    ```

3. Create a tpu cluster (from my experiments, n2-standard-2 has enough memory & storage but you might need more)

    ```shell
    gcloud compute tpus execution-groups create \
    --name="${TPU}" \
    --zone="${ZONE}" \
    --tf-version=2.4.1 \
    --machine-type=n2-standard-2 \
    --accelerator-type=v3-8
    ```

4. It should automatically connect to your TPU. If not:

    ```shell
    gcloud ssh $TPU --zone $ZONE
    ```

5. Download the Indirect Response code

    ```shell
    git clone https://github.com/frederiknolte/indirect-response
    cd indirect_response
    git checkout allennllp-t5
    git submodule init
    git submodule update
    ```

6. Download and install packages

    ```shell
    ./gcp_setup.sh
    cd google-research/wt5
    ```

7. Prepare your mixture task. This is an example:

    ```shell
    export TASK=your-chosen-mixture-task
    export PRETRAINED_DIR=gs://t5-data/pretrained_models/large
    export PRETRAINED_STEPS=1000700
    export FINETUNE_STEPS=20000
    export MODEL_DIR="${BUCKET}/${TASK}"
    ```

8. Execute your mixture task to finetune the model. This is just an example:

    ```shell
    PYTHONPATH=$PYTHONPATH:/home/michaelneely/google-research/ t5_mesh_transformer \
    --tpu="${TPU}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
    --gin_file="wt5/gin/sequence_lengths/cos_e_v001.gin" \
    --gin_file="wt5/gin/sequence_lengths/esnli_v002.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="MIXTURE_NAME = '${TASK}'" \
    --gin_param="mesh_train_dataset_fn.use_cached=False" \
    --gin_param="utils.run.save_checkpoints_steps=2000" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
    --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
    --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.constant_learning_rate" \
    --gin_param="constant_learning_rate.learning_rate=1e-3" \
    --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
    --module_import="wt5.tasks" \
    --module_import="wt5.mixtures" \
    --module_import="circa.circa" \
    --gin_location_prefix="wt5/wt5/gin/"
    ```

9. Run your evaluation task

    ```shell
    TBD
    ```
