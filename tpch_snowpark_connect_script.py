"""
Snowflake Snowpark Connect for Apache Spark Job: TPCH Data Engineering Pipeline
Demonstrates enterprise Spark patterns: complex joins, window functions,
aggregations, CDC operations, and incremental processing.

Converts TPCH tables into curated dimension/fact tables with CDC support.
This script runs Spark workloads on Snowflake using Snowpark Connect for Apache Spark.
"""

# ============================================================================
# IMPORTS - Modified from Glue version
# Removed: awsglue.transforms, awsglue.utils, awsglue.context, awsglue.job
# Added: Standard SparkSession with Snowpark Connect instead of GlueContext
# ============================================================================
import sys
import os
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import DataFrame
import snowflake.snowpark
from snowflake import snowpark_connect

# ============================================================================
# Snowpark Connect for Apache Spark Configuration
# NEW - Snowflake-specific connection configuration (not in Glue version)
# ============================================================================
os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"
snowpark_connect.start_session()  # Start the local Snowpark Connect for Spark session
spark = snowpark_connect.get_session()
spark.conf.set("snowpark.connect.sql.identifiers.auto-uppercase", "none")

print(f"✅ Spark Session created: {spark.version}")

# Optimize Spark for disk-constrained environments
# Reduce shuffle partitions to minimize disk spill
spark.conf.set("spark.sql.shuffle.partitions", "50")

# Configuration - Database and catalog information
DATABASE = 'tosmith_iceberg_project_db'
CATALOG_PREFIX = 'AWS_GLUE_CATALOG_LINKED_DB'  # Snowflake catalog linked to AWS Glue

# Configurable parameters (with defaults)
# NOTE: If you hit "No space left on device", reduce CDC_SAMPLE_SIZE to 5000 or lower
CDC_SAMPLE_SIZE = int(os.environ.get('CDC_SAMPLE_SIZE', '5000'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))

# ============================================================================
# TIMING INFRASTRUCTURE - Same as original Glue Spark code
# ============================================================================
step_timings = []
job_start_time = time.time()

def log_step_start(step_name):
    """Log the start of a processing step"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*80}")
    print(f"⏱️  [{timestamp}] Starting: {step_name}")
    print(f"{'='*80}")
    return time.time()

def log_step_end(step_name, start_time):
    """Log the end of a processing step and record duration"""
    end_time = time.time()
    duration = end_time - start_time
    step_timings.append((step_name, duration))
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"✅ [{timestamp}] Completed: {step_name} (Duration: {duration:.2f}s)")
    return end_time

print("="*80)
print("TPCH DATA ENGINEERING PIPELINE - ICEBERG")
print("Running on Snowflake with Snowpark Connect for Apache Spark")
print(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"Database: {DATABASE}")
print(f"Catalog: {CATALOG_PREFIX}")
print(f"CDC Sample Size: {CDC_SAMPLE_SIZE:,}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

# ============================================================================
# Helper Functions - Modified for Snowpark Connect
# ============================================================================
def df_lower_columns(df: DataFrame) -> DataFrame:
    """Convert all column names to lowercase."""
    return df.toDF(*[c.lower() for c in df.columns])

def read_iceberg_table(table: str) -> DataFrame:
    """Read an Iceberg table from AWS Glue Catalog via Snowflake."""
    full_table_name = f"{CATALOG_PREFIX}.{DATABASE}.{table}"
    return spark.table(full_table_name)

def write_iceberg_table(df: DataFrame, table: str, mode: str = "overwrite"):
    """Write a DataFrame to an Iceberg table using Snowpark Connect."""
    full_table_name = f"{CATALOG_PREFIX}.{DATABASE}.{table}"
    
    if mode == "overwrite":
        df.write \
            .format("iceberg") \
            .mode("overwrite") \
            .option("write.parquet.compression-codec", "snappy") \
            .saveAsTable(full_table_name)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'overwrite'.")

# # ============================================================================
# # STEP 0: Iceberg Write Test
# # ============================================================================
# step_start = log_step_start("Iceberg Write Test")
# try:
#     test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
#     write_iceberg_table(test_df, "_write_test")
    
#     # Verify we can read it back
#     test_read = read_iceberg_table("_write_test")
#     test_count = test_read.count()
#     print(f"   Read {test_count} rows from test table")
    
#     # Clean up test table
#     spark.sql(f"DROP TABLE IF EXISTS {CATALOG_PREFIX}.{DATABASE}._write_test")
#     print("✅ Write test PASSED - Iceberg configuration is working!")
#     log_step_end("Iceberg Write Test", step_start)
#     print("🎯 STAGE COMPLETE: Iceberg Write Test\n")
# except Exception as e:
#     print("❌ Write test FAILED - Cannot write Iceberg tables!")
#     print(f"Error: {e}")
#     raise RuntimeError(f"Iceberg write test failed. Fix configuration before running pipeline: {e}")

# ============================================================================
# STEP 1: Read TPCH Source Iceberg Tables
# MODIFIED - Different read method from Glue version
# Glue used: glueContext.create_data_frame.from_catalog()
# Snowflake uses: spark.table() to read from AWS Glue Catalog
# ============================================================================
step_start = log_step_start("Step 1: Read TPCH Source Tables")

required_tables = [
    "customer", "orders", "lineitem",
    "part", "supplier", "partsupp",
    "nation", "region"
]

dfs = {}
for table_name in required_tables:
    try:
        print(f"   Reading: {CATALOG_PREFIX}.{DATABASE}.{table_name}")
        
        # Read using spark.table() for Snowpark Connect
        df = read_iceberg_table(table_name)
        
        df = df_lower_columns(df)
        dfs[table_name] = df
        print(f"   ✅ {table_name}: {df.count():,} rows")
    except Exception as e:
        print(f"   ⚠️  Could not read {table_name}: {e}")
        # Continue with available tables

print(f"\n   Successfully loaded {len(dfs)} tables")
log_step_end("Step 1: Read TPCH Source Tables", step_start)
print("🎯 STAGE COMPLETE: Step 1 - Read TPCH Source Tables\n")

# ============================================================================
# STEP 2: Customer Geography Enrichment
# Same as original Glue Spark code
# ============================================================================
step_start = log_step_start("Step 2: Customer Geography Enrichment")

customer = dfs["customer"]
nation = dfs["nation"]
region = dfs["region"]

cust_geo = (
    customer.alias("c")
    .join(nation.alias("n"), col("c.c_nationkey") == col("n.n_nationkey"), "left")
    .join(region.alias("r"), col("n.n_regionkey") == col("r.r_regionkey"), "left")
    .select(
        col("c.c_custkey").alias("custkey"),
        col("c.c_name").alias("name"),
        col("c.c_acctbal").alias("acctbal"),
        col("c.c_mktsegment").alias("mktsegment"),
        col("n.n_name").alias("nation"),
        col("r.r_name").alias("region")
    )
)

cust_geo_count = cust_geo.count()
print(f"   Customer geography records: {cust_geo_count:,}")
log_step_end("Step 2: Customer Geography Enrichment", step_start)
print("🎯 STAGE COMPLETE: Step 2 - Customer Geography Enrichment\n")

# ============================================================================
# STEP 3: Order Revenue Calculation
# Same as original Glue Spark code
# ============================================================================
step_start = log_step_start("Step 3: Order Revenue Calculation")

lineitem = dfs["lineitem"]
orders = dfs["orders"]

order_revenue = (
    lineitem.groupBy("l_orderkey")
    .agg(sum(col("l_extendedprice") * (1 - col("l_discount"))).alias("order_revenue"))
)

orders_enriched = (
    orders.alias("o")
    .join(order_revenue.alias("rev"), col("o.o_orderkey") == col("rev.l_orderkey"), "left")
    .select(
        col("o.o_orderkey").alias("orderkey"),
        col("o.o_custkey").alias("custkey"),
        col("o.o_orderstatus").alias("orderstatus"),
        col("o.o_totalprice").alias("totalprice"),
        col("o.o_orderdate").alias("orderdate"),
        col("o.o_orderpriority").alias("orderpriority"),
        col("o.o_clerk").alias("clerk"),
        col("o.o_shippriority").alias("shippriority"),
        col("rev.order_revenue").alias("order_revenue")
    )
)

orders_count = orders_enriched.count()
print(f"   Orders with revenue: {orders_count:,}")
log_step_end("Step 3: Order Revenue Calculation", step_start)
print("🎯 STAGE COMPLETE: Step 3 - Order Revenue Calculation\n")

# ============================================================================
# STEP 4: Window Functions - Rankings & Rolling Aggregates
# Same as original Glue Spark code
# ============================================================================
step_start = log_step_start("Step 4: Window Functions")

w_latest = Window.partitionBy("custkey").orderBy(col("orderdate").desc())
w_revenue_rank = Window.partitionBy("custkey").orderBy(col("order_revenue").desc_nulls_last())

orders_w = (
    orders_enriched
    .withColumn("rn_latest", row_number().over(w_latest))
    .withColumn("revenue_rank", dense_rank().over(w_revenue_rank))
)

# Rolling 30-day revenue window
w_rolling = (
    Window.partitionBy("custkey")
    .orderBy(col("orderdate").cast("timestamp").cast("long"))
    .rangeBetween(-30 * 24 * 3600, 0)
)

orders_w = orders_w.withColumn(
    "rolling_30d_revenue",
    sum("order_revenue").over(w_rolling)
)

windowed_count = orders_w.count()
print(f"   Window functions applied: {windowed_count:,} rows")
log_step_end("Step 4: Window Functions", step_start)
print("🎯 STAGE COMPLETE: Step 4 - Window Functions\n")

# ============================================================================
# STEP 5: Create Dimension Table - dim_customer
# ============================================================================
step_start = log_step_start("Step 5: Create dim_customer")

dim_customer_df = cust_geo.dropDuplicates(["custkey"])

write_iceberg_table(dim_customer_df, "dim_customer")

dim_count = dim_customer_df.count()
print(f"   dim_customer created: {dim_count:,} rows")
log_step_end("Step 5: Create dim_customer", step_start)
print("🎯 STAGE COMPLETE: Step 5 - Create dim_customer\n")

# ============================================================================
# STEP 6: Create Fact Table - fact_orders
# ============================================================================
step_start = log_step_start("Step 6: Create fact_orders")

fact_orders_df = orders_w.select(
    "orderkey", "custkey", "orderstatus", "totalprice", "orderdate",
    "orderpriority", "clerk", "shippriority", "order_revenue",
    "rn_latest", "revenue_rank", "rolling_30d_revenue"
)

write_iceberg_table(fact_orders_df, "fact_orders")

fact_count = fact_orders_df.count()
print(f"   fact_orders created: {fact_count:,} rows")
log_step_end("Step 6: Create fact_orders", step_start)
print("🎯 STAGE COMPLETE: Step 6 - Create fact_orders\n")

# ============================================================================
# STEP 7: Create Aggregate Table - agg_customer_revenue
# ============================================================================
step_start = log_step_start("Step 7: Create agg_customer_revenue")

agg_revenue_df = (
    fact_orders_df.groupBy("custkey")
    .agg(
        countDistinct("orderkey").alias("order_cnt"),
        sum("order_revenue").alias("lifetime_revenue"),
        max("orderdate").alias("last_order_date")
    )
)

write_iceberg_table(agg_revenue_df, "agg_customer_revenue")

agg_count = agg_revenue_df.count()
print(f"   agg_customer_revenue created: {agg_count:,} rows")
log_step_end("Step 7: Create agg_customer_revenue", step_start)
print("🎯 STAGE COMPLETE: Step 7 - Create agg_customer_revenue\n")

print("\n✅ Created/overwritten target tables:")
print(f"   - {CATALOG_PREFIX}.{DATABASE}.dim_customer")
print(f"   - {CATALOG_PREFIX}.{DATABASE}.fact_orders")
print(f"   - {CATALOG_PREFIX}.{DATABASE}.agg_customer_revenue")

# ============================================================================
# STEP 8: Simulate CDC Operations
# Same as original Glue Spark code
# ============================================================================
step_start = log_step_start("Step 8: Simulate CDC Operations")

# Sample random orders for CDC simulation
sim_src = (
    orders_enriched
    .withColumn("rnd", rand(seed=RANDOM_SEED))
    .withColumn("rn", row_number().over(Window.orderBy("rnd")))
    .filter(col("rn") <= CDC_SAMPLE_SIZE)
    .drop("rnd", "rn")
)

# Assign random operations: 30% Insert, 30% Delete, 40% Update
sim_cdc = (
    sim_src
    .withColumn("p", rand(seed=RANDOM_SEED))
    .withColumn(
        "op",
        when(col("p") < 0.30, lit("I"))
         .when(col("p") < 0.60, lit("D"))
         .otherwise(lit("U"))
    )
    .drop("p")
)

# For inserts, offset the orderkey to avoid conflicts
INSERT_KEY_OFFSET = 10_000_000
sim_cdc = sim_cdc.withColumn(
    "merge_orderkey",
    when(col("op") == "I", col("orderkey") + lit(INSERT_KEY_OFFSET)).otherwise(col("orderkey"))
)

# For updates, modify orderpriority to show the change
sim_cdc = sim_cdc.withColumn(
    "orderpriority",
    when(col("op") == "U", concat_ws("-", col("orderpriority"), lit("upd"))).otherwise(col("orderpriority"))
)

sim_cdc = sim_cdc.select(
    col("merge_orderkey").alias("orderkey"),
    "custkey", "orderstatus", "totalprice", "orderdate",
    "orderpriority", "clerk", "shippriority", "order_revenue", "op"
)

print("   CDC operation breakdown (before deduplication):")
sim_cdc.groupBy("op").count().show()

# CRITICAL: Deduplicate by orderkey BEFORE any further processing
# This ensures the MERGE source is deterministic
print("   Deduplicating CDC data by orderkey (keeping last operation per key)...")
sim_cdc = sim_cdc.dropDuplicates(["orderkey"])
print("   ✅ CDC data deduplicated")

log_step_end("Step 8: Simulate CDC Operations", step_start)
print("🎯 STAGE COMPLETE: Step 8 - Simulate CDC Operations\n")

# ============================================================================
# STEP 9: Apply CDC via Iceberg MERGE
# ============================================================================
step_start = log_step_start("Step 9: Apply CDC via MERGE")

# Prepare CDC data with proper schema alignment
cdc_data = (
    sim_cdc
    .withColumn("rn_latest", lit(0).cast("int"))
    .withColumn("revenue_rank", lit(0).cast("int"))
    .withColumn("rolling_30d_revenue", lit(0.0).cast("double"))
    .withColumn("orderstatus", substring(col("orderstatus"), 1, 10))
    .withColumn("orderpriority", substring(col("orderpriority"), 1, 15))
    .withColumn("clerk", substring(col("clerk"), 1, 15))
)

# ============================================================================
# CRITICAL FIX: Break non-deterministic lineage by overwriting persistent table
# ============================================================================
# The rand() functions used earlier create a non-deterministic lineage that
# Spark's MERGE operator rejects. In Snowpark Connect, we can't CREATE new
# tables in catalog-linked databases, but we CAN overwrite existing ones.
# 
# IMPORTANT: The AWS Glue script must run first to create the _temp_cdc_source
# table. Then this script can overwrite it with new CDC data.

temp_cdc_table = f"{CATALOG_PREFIX}.{DATABASE}._temp_cdc_source"
print(f"   Materializing CDC data by overwriting persistent temp table...")

# Overwrite the existing temp CDC table (must exist from prior Glue run)
try:
    cdc_data.write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable(temp_cdc_table)
    print(f"   ✅ Overwrote existing temp table: {temp_cdc_table}")
except Exception as e:
    print(f"   ❌ Failed to overwrite temp table. Did you run the Glue script first?")
    print(f"   Error: {e}")
    raise RuntimeError(f"Temp CDC table {temp_cdc_table} must exist. Run AWS Glue script first to create it.")

# Read back from temp table - this creates a clean, deterministic source
cdc_source_clean = spark.table(temp_cdc_table)
cdc_data_count = cdc_source_clean.count()
print(f"   ✅ Read {cdc_data_count:,} CDC rows from persistent temp table")

# Get target data columns (excluding 'op')
target_data_columns = [c for c in cdc_source_clean.columns if c != 'op']

# Generate column assignments for UPDATE and INSERT
update_set_clause = ", ".join([f"target.{c} = source.{c}" for c in target_data_columns])
insert_cols_clause = f"({', '.join(target_data_columns)})"
insert_vals_clause = f"({', '.join([f'source.{c}' for c in target_data_columns])})"

# Create temporary view for CDC data from the clean source
cdc_source_clean.createOrReplaceTempView("cdc_source")
print("   ✅ Created deterministic CDC source view for MERGE")

# Execute MERGE INTO statement
full_fact_table = f"{CATALOG_PREFIX}.{DATABASE}.fact_orders"
merge_sql = f"""
MERGE INTO {full_fact_table} AS target
USING cdc_source AS source
ON target.orderkey = source.orderkey
WHEN MATCHED AND source.op = 'D' THEN
    DELETE
WHEN MATCHED AND source.op = 'U' THEN
    UPDATE SET {update_set_clause}
WHEN NOT MATCHED AND source.op = 'I' THEN
    INSERT {insert_cols_clause} VALUES {insert_vals_clause}
"""

try:
    print("   Executing MERGE statement...")
    spark.sql(merge_sql)
    print("   ✅ CDC operations completed successfully using MERGE!")
except Exception as e:
    print(f"   ❌ MERGE failed: {e}")
    raise

# NOTE: We keep the temp CDC table persistent (don't drop it) for reuse
# on subsequent runs. The table will be overwritten each time.
print(f"   ℹ️  Keeping {temp_cdc_table} persistent for future runs")

# Post-CDC verification
row_count_after_cdc = spark.sql(f"SELECT COUNT(*) FROM {full_fact_table}").collect()[0][0]
print(f"   Rows in fact_orders after CDC: {row_count_after_cdc:,}")
log_step_end("Step 9: Apply CDC via MERGE", step_start)
print("🎯 STAGE COMPLETE: Step 9 - Apply CDC via MERGE\n")

# ============================================================================
# STEP 10: Refresh Aggregates After CDC
# ============================================================================
step_start = log_step_start("Step 10: Refresh Aggregates")

fact_orders_post = spark.sql(f"SELECT * FROM {full_fact_table}")

agg_revenue_post = (
    fact_orders_post.groupBy("custkey")
    .agg(
        countDistinct("orderkey").alias("order_cnt"),
        sum("order_revenue").alias("lifetime_revenue"),
        max("orderdate").alias("last_order_date")
    )
)

write_iceberg_table(agg_revenue_post, "agg_customer_revenue")
print(f"   ✅ Refreshed: {CATALOG_PREFIX}.{DATABASE}.agg_customer_revenue")

agg_post_count = agg_revenue_post.count()
print(f"   Updated aggregate rows: {agg_post_count:,}")
log_step_end("Step 10: Refresh Aggregates", step_start)
print("🎯 STAGE COMPLETE: Step 10 - Refresh Aggregates\n")

# ============================================================================
# SUMMARY - Same as original Glue Spark code with Snowflake mentions
# ============================================================================
job_end_time = time.time()
total_job_duration = job_end_time - job_start_time

print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)
print(f"Job completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total job duration: {total_job_duration:.2f}s ({total_job_duration/60:.2f} minutes)")
print("="*80)
print(f"\nSource tables processed: {len(dfs)}")
print(f"Target tables created: 3 (dimension, fact, aggregate)")
print(f"CDC operations simulated: {CDC_SAMPLE_SIZE:,}")
print("\nTransformations applied:")
print("  ✅ Customer geography enrichment")
print("  ✅ Order revenue calculation")
print("  ✅ Window functions (rankings & rolling aggregates)")
print("  ✅ Dimension & fact table creation")
print("  ✅ CDC operations (Insert/Update/Delete)")
print("  ✅ Aggregate refresh")

print("\n" + "="*80)
print("PERFORMANCE BREAKDOWN (by step)")
print("="*80)
for step_name, duration in step_timings:
    percentage = (duration / total_job_duration) * 100 if total_job_duration > 0 else 0
    print(f"{step_name:.<60} {duration:>8.2f}s ({percentage:>5.1f}%)")
print("="*80)
print(f"{'TOTAL':.<60} {total_job_duration:>8.2f}s (100.0%)")
print("="*80)

# Stop Spark session - Snowflake-specific cleanup
spark.stop()
print("\n✅ Spark session stopped successfully")
print("="*80)
