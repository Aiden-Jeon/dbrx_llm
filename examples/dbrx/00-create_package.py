# Databricks notebook source
# MAGIC %md
# MAGIC # Create Pacakge

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.package")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Build pacakge

# COMMAND ----------

# MAGIC %sh
# MAGIC make -C ../dbrx_llm/ build

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Copy to volume

# COMMAND ----------

import os

dist_path = "file:" + os.path.abspath(os.path.join(os.getcwd(), "../dbrx_llm/dist"))
dest_path = f"/Volumes/{catalog}/{schema}/package"
print(f"copy dist from {dist_path} to {dest_path}")
dbutils.fs.cp(
    dist_path,
    dest_path,
    recurse=True,
)

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime

files_df = spark.createDataFrame(dbutils.fs.ls(dest_path))
files_df = files_df.withColumn("modificationTime", from_unixtime(col("modificationTime") / 1000))
display(files_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Install and validate

# COMMAND ----------

# MAGIC %pip install /Volumes/jongseob_demo/distributed/package/dbrx_llm-0.1.0-py3-none-any.whl

# COMMAND ----------

import dbrx_llm

dbrx_llm.__version__