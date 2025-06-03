import os


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9340"
os.environ["RANK"] = str(0)
os.environ["LOCAL_RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)


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