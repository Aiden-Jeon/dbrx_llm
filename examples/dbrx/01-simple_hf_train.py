# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Code for Training Huggingface Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Custom Package

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
# MAGIC ## Train on single node
# MAGIC

# COMMAND ----------

def train_single_node(
    model_name,
    dataset_name,
    log_dir,
    experiment_path,
    num_epochs=1,
    batch_size=16,
    log_interval=10,
):
    import mlflow
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from dbrx_llm.models import SentimentClassifier
    from dbrx_llm.dataset import load_torch_dataset
    from dbrx_llm.train import train_one_epoch
    from dbrx_llm.utils import save_checkpoint
    from dbrx_llm.eval import AverageMeter, evaluate_one_epoch

    #################### Setting up MLflow ####################
    experiment = mlflow.set_experiment(experiment_path)
    ###########################################################

    ###########################################################
    #################### Setting up dataset ###################
    tokenized_datasets = load_torch_dataset(dataset_name, model_name)
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ###########################################################
    ##################### Setting up model ####################
    device = torch.device("cuda")
    model = SentimentClassifier(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    ###########################################################

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            max_duration=10,
        )
        save_checkpoint(log_dir, model, epoch)

    print("Testing...")
    avg_meter = AverageMeter()
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    evaluate_one_epoch(
        model=model,
        data_loader=test_dataloader,
        avg_meter=avg_meter,
        log_interval=log_interval,
        max_duration=10,
    )
    test_loss, test_acc = avg_meter.reduce()

    print("Average test loss: {}, accuracy: {}".format(test_loss, test_acc))
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    return "Finished"


# COMMAND ----------

import mlflow
from dbrx_llm.utils import create_log_dir

args = Arguments()
log_dir = create_log_dir(args.log_volume_dir)
print("Log directory:", log_dir)
with mlflow.start_run():
    train_single_node(
        # meta
        log_dir=log_dir,
        experiment_path=args.experiment_path,
        # model
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        # param
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

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
    model_name,
    dataset_name,
    log_dir,
    experiment_path,
    num_epochs=1,
    batch_size=16,
    log_interval=10,
):
    #### Added imports here ####
    import mlflow
    import torch.optim as optim
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    from dbrx_llm.models import SentimentClassifier
    from dbrx_llm.dataset import load_torch_dataset
    from dbrx_llm.train import train_one_epoch
    from dbrx_llm.utils import save_checkpoint
    from dbrx_llm.eval import AverageMeter, evaluate_one_epoch

    ###########################################################
    #################### Setting up MLflow ####################
    # We need to do this so that different processes that will be able to find mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    # We set the experiment details here
    experiment = mlflow.set_experiment(experiment_path)
    ###########################################################

    print("Running distributed training")
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    ###########################################################
    #################### Setting up dataset ###################
    tokenized_datasets = load_torch_dataset(dataset_name, model_name)
    train_dataset = tokenized_datasets["train"]

    ############## Added Distributed Dataloader ###############
    train_sampler = DistributedSampler(dataset=train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    ################## Added Distributed Model ################
    model = SentimentClassifier(model_name).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ddp_optimizer = optim.AdamW(ddp_model.parameters(), lr=5e-5)
    ###########################################################

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model=ddp_model,
            optimizer=ddp_optimizer,
            data_loader=train_dataloader,
            device=local_rank,
            epoch=epoch,
            log_interval=log_interval,
            max_duration=10,
        )
        if global_rank == 0:
            save_checkpoint(log_dir, model, epoch)
    dist.barrier()

    print("Testing...")
    avg_meter = AverageMeter()
    test_dataset = tokenized_datasets["test"]
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler
    )
    evaluate_one_epoch(
        model=ddp_model,
        data_loader=test_dataloader,
        avg_meter=avg_meter,
        log_interval=log_interval,
        max_duration=10,
    )
    dist.barrier()
    test_loss, test_acc = avg_meter.all_reduce("cuda")
    # save out the model for test
    if global_rank == 0:
        print("Average test loss: {}, accuracy: {}".format(test_loss, test_acc))
        # mlflow.pytorch.log_model(ddp_model, "model")
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
    dist.barrier()
    dist.destroy_process_group()

    return "Finished"  # can return any picklable object

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor
from dbrx_llm.utils import create_log_dir

args = Arguments()
log_dir = create_log_dir(args.log_volume_dir)
print("Log directory:", log_dir)

output_dist = TorchDistributor(
    num_processes=4,
    local_mode=False,
    use_gpu=True,
).run(
    train_multi_nodes,
    # meta
    log_dir=log_dir,
    experiment_path=args.experiment_path,
    # model
    model_name=args.model_name,
    dataset_name=args.dataset_name,
    # param
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    log_interval=args.log_interval,
)