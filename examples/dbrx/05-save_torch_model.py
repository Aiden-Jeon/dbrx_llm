# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Code for Training Huggingface Model

# COMMAND ----------

# MAGIC %md
# MAGIC # Simple Code for Training Huggingface Model

# COMMAND ----------

# MAGIC %pip install -q /Volumes/jongseob_demo/distributed/package/dbrx_llm-0.1.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configs

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"USE {catalog}.{schema}")
spark.sql("CREATE VOLUME IF NOT EXISTS log_dir")
spark.sql("CREATE VOLUME IF NOT EXISTS autoresume")

# COMMAND ----------

import os
from dataclasses import dataclass


@dataclass
class Arguments:
    # meta
    log_volume_dir = f"/Volumes/{catalog}/{schema}/log_dir"
    experiment_path = os.path.join(os.getcwd().replace("/Workspace", ""), "experiments")
    save_volume_dir = f"/Volumes/{catalog}/{schema}/autoresume"
    
    # data streaming
    out_root = f"/Volumes/{catalog}/{schema}/mds/mds-text/"
    train_data_path = os.path.join(out_root, "spark_train")
    test_data_path = os.path.join(out_root, "spark_test")

    # model
    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"

    # params
    log_interval = 10
    batch_size = 16
    num_epochs = 1

# COMMAND ----------

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply("api_url")
db_token = context.apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Train on multi node

# COMMAND ----------

import os


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9340"
os.environ["RANK"] = str(0)
os.environ["LOCAL_RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)


# For distributed training we will merge the train and test steps into 1 main function
def train_multi_nodes(
    postfix,
    model_name,
    train_data_path,
    test_data_path,
    experiment_path,
    save_folder,
    save_interval=1,
    autoresume=True,
    num_epochs=1,
    batch_size=16,
):
    #### Added imports here ####
    import shutil
    import mlflow
    import composer
    import torch.optim as optim
    import torch.distributed as dist
    import streaming.base.util as util

    from composer import Trainer
    from composer.utils import get_device

    from dbrx_llm.models import ComposerSentimentClassifier
    from dbrx_llm.dataset import get_mds_dataloader

    ###########################################################
    #################### Setting up MLflow ####################
    # set environment variables for Databricks and TMPDIR for mlflow_logger
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    print("Creating mlflow logger..........\n")
    experiment = mlflow.set_experiment(experiment_path)
    mlflow_logger = composer.loggers.MLFlowLogger(
        experiment_name=experiment_path,
        synchronous=True,
        resume=True,
    )
    run_name = f"my_auto_resume-{postfix}"
    ###########################################################

    print("Running distributed training")
    dist.init_process_group("nccl")

    # mosaic streaming recommendations
    util.clean_stale_shared_memory()
    composer.utils.dist.initialize_dist(get_device(None))

    ###########################################################
    #################### Setting up dataset ###################
    train_dataloader = get_mds_dataloader(
        train_data_path, batch_size, "train", use_local=False
    )
    test_dataloader = get_mds_dataloader(
        test_data_path, batch_size, "test", use_local=False
    )
    ################## Added Distributed Model ################
    model = ComposerSentimentClassifier(model_name)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    ###########################################################

    print("Copy model from volume to local\n")
    local_save_folder = f"/local_disk0/autoresume/{postfix}"
    os.makedirs(local_save_folder, exist_ok=True)
    if os.path.exists(save_folder):
        shutil.copytree(save_folder, local_save_folder, dirs_exist_ok=True)

    dist.barrier()
    # Create Trainer Object
    print("Creating Composer Trainer\n")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        optimizers=optimizer,
        device="gpu",
        max_duration=num_epochs,
        loggers=[mlflow_logger],
        save_folder=local_save_folder,
        save_interval=save_interval,
        autoresume=autoresume,
        run_name=run_name,
    )
    # Start training
    print("Starting training\n")
    trainer.fit()
    trainer.close()
    mlflow.end_run()

    if int(os.environ["RANK"]) == 0:
        print("Copy model from local to volume\n")
        shutil.copytree(local_save_folder, save_folder, dirs_exist_ok=True)

    dist.barrier()
    dist.destroy_process_group()
    return model


# COMMAND ----------

from time import time

import mlflow
from pyspark.ml.torch.distributor import TorchDistributor

args = Arguments()
postfix = str(time())
save_volume_path = os.path.join(args.save_volume_dir, postfix)
output_model = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True,
).run(
    train_multi_nodes,
    postfix=postfix,
    model_name=args.model_name,
    train_data_path=args.train_data_path,
    test_data_path=args.test_data_path,
    experiment_path=args.experiment_path,
    save_folder=save_volume_path,
    num_epochs="10ba",
    batch_size=args.batch_size,
)

# COMMAND ----------

from dbrx_llm.models import PyfuncSentimentClassifier
from dbrx_llm.dataset import get_mds_dataloader
from mlflow.utils.environment import _mlflow_conda_env

run_name = f"my_auto_resume-{postfix}"

experiment = mlflow.set_experiment(args.experiment_path)
experiment_id = experiment.experiment_id
run_id = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.run_name='{run_name}'",
)["run_id"][0]
train_dataloader = get_mds_dataloader(
    args.train_data_path, args.batch_size, "train", use_local=False
)

batch = next(iter(train_dataloader))
input_example = {
    "input_ids": batch["input_ids"][0].numpy(),
    "attention_mask": batch["attention_mask"][0].numpy(),
}
print("Save model to mlflow\n")
with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
    batch = next(iter(train_dataloader))
    input_example = {
        "input_ids": batch["input_ids"][0].numpy(),
        "attention_mask": batch["attention_mask"][0].numpy(),
    }
    custom_model = PyfuncSentimentClassifier(output_model.model.to("cuda"))
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "torch==2.3.1",
            "transformers==4.41.2",
            "/Volumes/jongseob_demo/distributed/package/dbrx_llm-0.1.0-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=custom_model,
        artifact_path="model",
        input_example=input_example,
        conda_env=conda_env,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Regsiter Model

# COMMAND ----------

import mlflow

mlflow.register_model(
    name=f"{catalog}.{schema}.pytorch_classifier",
    model_uri=f"runs:/{run_id}/model",
)

# COMMAND ----------

import mlflow.models.utils

mlflow.models.utils.add_libraries_to_model(
    f"models:/{catalog}.{schema}.pytorch_classifier/5",
)

# COMMAND ----------



# COMMAND ----------

