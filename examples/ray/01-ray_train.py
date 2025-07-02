# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Code for Training Huggingface Model

# COMMAND ----------

# MAGIC %pip install -q /Volumes/jongseob_demo/distributed/package/dbrx_llm-0.1.0-py3-none-any.whl
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"USE {catalog}.{schema}")
spark.sql("CREATE VOLUME IF NOT EXISTS log_dir")

# COMMAND ----------

import os
from dataclasses import dataclass


@dataclass
class Arguments:
    # meta
    log_volume_dir = f"/Volumes/{catalog}/{schema}/log_dir"
    experiment_path = os.path.join(os.getcwd().replace("/Workspace", ""), "experiments")

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
# MAGIC ## Train Code

# COMMAND ----------

import os
from typing import Dict

import ray
import mlflow
import torch
import torch.optim as optim
from ray.air.integrations.mlflow import setup_mlflow
from mlflow.utils.databricks_utils import get_databricks_env_vars
from torch.utils.data import DataLoader

from dbrx_llm.models import SentimentClassifier
from dbrx_llm.dataset import load_torch_dataset
from dbrx_llm.train import train_one_epoch
from dbrx_llm.eval import AverageMeter, evaluate_one_epoch

mlflow_db_creds = get_databricks_env_vars("databricks")

def train_func_per_worker(config: Dict):
    experiment_path = config["experiment_path"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    log_interval = config["log_interval"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    #################### Setting up MLflow ####################
    print("Set MLflow logger")
    # We need to do this so that different processes that will be able to find mlflow
    os.environ.update(mlflow_db_creds)
    mlflow.set_experiment(experiment_path)
    ###########################################################

    ###########################################################
    #################### Setting up dataset ###################
    # Get dataloaders inside the worker training function
    print("Get datasets")
    tokenized_datasets = load_torch_dataset(dataset_name, model_name)
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # [1] Prepare Dataloader for distributed training
    # Shard the datasets among workers and move batches to the correct device
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    ###########################################################
    ##################### Setting up model ####################
    # [2] Prepare and wrap your model with DistributedDataParallel
    # Move the model to the correct GPU/CPU device
    print("Set models")
    device = torch.device("cuda")
    model = SentimentClassifier(model_name).to(device)
    model = ray.train.torch.prepare_model(model)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    ###########################################################
    print("Start Training")

    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            train_dataloader.sampler.set_epoch(epoch)
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            max_duration=10,
        )
        # save_checkpoint(log_dir, model, epoch)
        avg_meter = AverageMeter()
        evaluate_one_epoch(
            model=model,
            data_loader=test_dataloader,
            avg_meter=avg_meter,
            log_interval=log_interval,
            max_duration=10,
        )
        test_loss, test_acc = avg_meter.all_reduce("cuda")
        # [3] Report metrics to Ray Train
        # ===============================
        ray.train.report(metrics={"loss": test_loss, "accuracy": test_acc})
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with single node

# COMMAND ----------

# Step 3: Create a TorchTrainer. Specify the number of training workers and
# pass in your Ray Dataset.
# The Ray Dataset is automatically split across all training workers.
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

args = Arguments()
train_config = {
    "experiment_path": args.experiment_path,
    "model_name": args.model_name,
    "dataset_name": args.dataset_name,
    "log_interval": args.log_interval,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
}
trainer = TorchTrainer(
    train_func_per_worker,
    train_loop_config=train_config,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
)

result = trainer.fit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train with multi node

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster

ray.util.spark.shutdown_ray_cluster()
setup_ray_cluster(max_worker_nodes=4)

# COMMAND ----------

# Step 3: Create a TorchTrainer. Specify the number of training workers and
# pass in your Ray Dataset.
# The Ray Dataset is automatically split across all training workers.
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

args = Arguments()
train_config = {
    "experiment_path": args.experiment_path,
    "model_name": args.model_name,
    "dataset_name": args.dataset_name,
    "log_interval": args.log_interval,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
}
trainer = TorchTrainer(
    train_func_per_worker,
    train_loop_config=train_config,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
result = trainer.fit()