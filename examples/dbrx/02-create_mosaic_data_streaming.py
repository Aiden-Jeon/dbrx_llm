# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic Data Streaming
# MAGIC

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
spark.sql("CREATE VOLUME IF NOT EXISTS mds")

# COMMAND ----------

import os

# Specify where the data will be stored
out_root = f"/Volumes/{catalog}/{schema}/mds/mds-text/"
output_dir_train = os.path.join(out_root, "spark_train")
output_dir_test = os.path.join(out_root, "spark_test")

# COMMAND ----------

# MAGIC %md ### 1. Convert Spark dataframe to MDS

# COMMAND ----------

from dbrx_llm.dataset import load_torch_dataset
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    LongType,
    ArrayType,
    StringType,
)

model_name = "distilbert-base-uncased"
dataset_name = "imdb"

tokenized_datasets = load_torch_dataset(dataset_name, model_name)
# Split dataset into train and validation sets
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Define schema
schema = StructType(
    [
        StructField("text", StringType(), True),
        StructField("label", LongType(), True),
        StructField("input_ids", ArrayType(IntegerType()), True),
        StructField("attention_mask", ArrayType(IntegerType()), True),
    ]
)

# Convert to Spark DataFrame
spark_train_dataset = spark.createDataFrame(train_dataset.to_pandas(), schema=schema)
spark_test_dataset = spark.createDataFrame(test_dataset.to_pandas(), schema=schema)

display(spark_train_dataset)

# COMMAND ----------

import os
from streaming.base.converters import dataframe_to_mds
from shutil import rmtree


# Parameters required for saving data in MDS format
columns = {
    "text": "str",
    "label": "int64",
    "input_ids": "ndarray:int32",
    "attention_mask": "ndarray:int32",
}

# compression algorithms
compression = "zstd:7"
hashes = ["sha1"]
limit = 8192


# Save the training data using the `dataframe_to_mds` function, which divides the dataframe into `num_workers` parts and merges the `index.json` from each part into one in a parent directory.
def save_data(df, output_path, label, num_workers=4):
    if os.path.exists(output_path):
        print(f"Deleting {label} data: {output_path}")
        rmtree(output_path)
    print(f"Saving {label} data to: {output_path}")
    mds_kwargs = {
        "out": output_path,
        "columns": columns,
        "compression": compression,
        "hashes": hashes,
        "size_limit": limit,
    }
    dataframe_to_mds(
        df.repartition(num_workers),
        merge_index=True,
        mds_kwargs=mds_kwargs,
    )


# save full dataset
save_data(spark_train_dataset, output_dir_train, "train", 10)
save_data(spark_test_dataset, output_dir_test, "test", 10)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Test mds dataloader

# COMMAND ----------

from dbrx_llm.dataset import get_mds_dataloader

train_dataloader = get_mds_dataloader(output_dir_train, batch_size=16, label="train")
for sample in train_dataloader:
    break

print(sample)
print(f"Total samples in MDS dataset: {len(train_dataloader)}")