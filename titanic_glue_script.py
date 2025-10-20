"""
AWS Glue Job: Complex Titanic Data Transformations
Demonstrates enterprise Spark patterns: window functions, feature engineering,
aggregations, cross joins, and ML-style transformations.

This will run for 2-5 minutes on Glue with proper complexity.
"""

import sys
import time
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Timing infrastructure
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
print("COMPLEX TITANIC TRANSFORMATIONS - ENTERPRISE PATTERNS")
print(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# WRITE TEST: Verify Iceberg write capability before processing
# ============================================================================
step_start = log_step_start("Iceberg Write Test")
try:
    test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
    test_df.createOrReplaceTempView("tmp_write_test")
    spark.sql("""
        CREATE OR REPLACE TABLE glue_catalog.tosmith_iceberg_project_db._write_test
        USING iceberg
        TBLPROPERTIES ('format-version'='2')
        AS SELECT * FROM tmp_write_test
    """)
    # Verify we can read it back
    spark.sql("SELECT * FROM glue_catalog.tosmith_iceberg_project_db._write_test").show()
    # Clean up test table
    spark.sql("DROP TABLE IF EXISTS glue_catalog.tosmith_iceberg_project_db._write_test")
    print("✅ Write test PASSED - Iceberg configuration is working!")
    log_step_end("Iceberg Write Test", step_start)
except Exception as e:
    print("❌ Write test FAILED - Cannot write Iceberg tables!")
    print(f"Error: {e}")
    raise RuntimeError(f"Iceberg write test failed. Fix configuration before running transformations: {e}")

# Debug: List available tables
print("\nDebug - Available tables in database:")
try:
    spark.sql("SHOW TABLES IN tosmith_iceberg_project_db").show()
except Exception as e:
    print(f"Could not list tables: {e}")

# ============================================================================
# STEP 1: Read Iceberg Data
# ============================================================================
step_start = log_step_start("Step 1: Read Iceberg Data")
# Use GlueContext method for reading Iceberg tables (AWS recommended)
df = glueContext.create_data_frame.from_catalog(
    database="tosmith_iceberg_project_db",
    table_name="titanic_test"
)

# Cast numeric columns properly
df = df.withColumn("AGE", col("AGE").cast("double")) \
       .withColumn("FARE", col("FARE").cast("double")) \
       .withColumn("SURVIVED", col("SURVIVED").cast("int")) \
       .withColumn("PCLASS", col("PCLASS").cast("int")) \
       .withColumn("SIBSP", col("SIBSP").cast("int")) \
       .withColumn("PARCH", col("PARCH").cast("int"))

row_count = df.count()
print(f"   Original rows: {row_count}")
log_step_end("Step 1: Read Iceberg Data", step_start)

# ============================================================================
# STEP 2: Data Explosion - Create Synthetic Complexity
# ============================================================================
step_start = log_step_start("Step 2: Data Explosion (Cross Join)")

# Cross-join to create comparison dataset (891 x 891 = 793,881 rows)
df_for_comparison = df.select(
    col("SURVIVED").alias("comp_survived"),
    col("PCLASS").alias("comp_pclass"),
    col("AGE").alias("comp_age"),
    col("FARE").alias("comp_fare"),
    col("SEX").alias("comp_sex")
)

# Create passenger pairs (simulates "network analysis" pattern in enterprise)
df_pairs = df.crossJoin(broadcast(df_for_comparison))
pairs_count = df_pairs.count()
print(f"   Pairs created: {pairs_count:,} rows")
log_step_end("Step 2: Data Explosion (Cross Join)", step_start)

# ============================================================================
# STEP 3: Complex Window Functions - Multiple Partitions & Orderings
# ============================================================================
step_start = log_step_start("Step 3: Window Functions")

# Define multiple window specifications (common in enterprise analytics)
window_class = Window.partitionBy("PCLASS").orderBy("FARE")
window_sex_class = Window.partitionBy("SEX", "PCLASS").orderBy("AGE")
window_embark = Window.partitionBy("EMBARKED").orderBy("FARE")
window_full = Window.partitionBy(lit(1)).orderBy("FARE")

# Calculate ranking metrics across different dimensions
df_windowed = df.withColumn("fare_rank_by_class", rank().over(window_class)) \
    .withColumn("fare_percentile_by_class", percent_rank().over(window_class)) \
    .withColumn("age_rank_by_sex_class", dense_rank().over(window_sex_class)) \
    .withColumn("fare_cumsum_by_embark", sum("FARE").over(window_embark.rowsBetween(Window.unboundedPreceding, Window.currentRow))) \
    .withColumn("age_moving_avg_3", avg("AGE").over(window_class.rowsBetween(-1, 1))) \
    .withColumn("fare_lag_1", lag("FARE", 1).over(window_class)) \
    .withColumn("fare_lead_1", lead("FARE", 1).over(window_class)) \
    .withColumn("global_fare_percentile", percent_rank().over(window_full))

windowed_count = df_windowed.count()
print(f"   Window functions applied: {windowed_count:,} rows")
log_step_end("Step 3: Window Functions", step_start)

# ============================================================================
# STEP 4: Feature Engineering - Complex Derived Features
# ============================================================================
step_start = log_step_start("Step 4: Feature Engineering")

# Family size features
df_features = df_windowed \
    .withColumn("family_size", col("SIBSP") + col("PARCH") + 1) \
    .withColumn("is_large_family", when(col("family_size") > 4, 1).otherwise(0)) \
    .withColumn("is_solo", when(col("ALONE") == "TRUE", 1).otherwise(0))

# Fare buckets and interactions
df_features = df_features \
    .withColumn("fare_bucket", 
        when(col("FARE") < 10, "low")
        .when(col("FARE") < 30, "medium")
        .when(col("FARE") < 100, "high")
        .otherwise("premium")) \
    .withColumn("age_bucket",
        when(col("AGE") < 12, "child")
        .when(col("AGE") < 18, "teen")
        .when(col("AGE") < 60, "adult")
        .otherwise("senior")) \
    .withColumn("fare_per_family_member", col("FARE") / col("family_size"))

# Complex boolean features
df_features = df_features \
    .withColumn("high_fare_solo", (col("fare_bucket") == "premium") & (col("is_solo") == 1)) \
    .withColumn("first_class_child", (col("PCLASS") == 1) & (col("age_bucket") == "child"))

print(f"   Features engineered: {len(df_features.columns)} columns total")
log_step_end("Step 4: Feature Engineering", step_start)

# ============================================================================
# STEP 5: Multiple Complex Aggregations with GroupBy Sets
# ============================================================================
step_start = log_step_start("Step 5: Multi-Dimensional Aggregations")

# Aggregation 1: Survival statistics by multiple dimensions
survival_stats = df_features.groupBy("PCLASS", "SEX", "age_bucket", "fare_bucket").agg(
    count("*").alias("passenger_count"),
    avg("SURVIVED").alias("survival_rate"),
    avg("FARE").alias("avg_fare"),
    stddev("FARE").alias("stddev_fare"),
    min("FARE").alias("min_fare"),
    max("FARE").alias("max_fare"),
    avg("AGE").alias("avg_age"),
    countDistinct("EMBARKED").alias("distinct_ports")
)

# Aggregation 2: Cohort analysis
cohort_analysis = df_features.groupBy("EMBARKED", "PCLASS").agg(
    sum("SURVIVED").alias("survivors"),
    count("*").alias("total_passengers"),
    avg("FARE").alias("avg_fare"),
    sum(when(col("SEX") == "MALE", 1).otherwise(0)).alias("male_count"),
    sum(when(col("SEX") == "FEMALE", 1).otherwise(0)).alias("female_count")
).withColumn("survival_rate", col("survivors") / col("total_passengers"))

# Aggregation 3: Family survival patterns
family_patterns = df_features.groupBy("family_size", "PCLASS").agg(
    avg("SURVIVED").alias("family_survival_rate"),
    count("*").alias("family_count"),
    avg("FARE").alias("avg_fare_paid")
)

survival_count = survival_stats.count()
cohort_count = cohort_analysis.count()
family_count = family_patterns.count()
print(f"   Survival stats groups: {survival_count:,}")
print(f"   Cohort analysis groups: {cohort_count:,}")
print(f"   Family patterns: {family_count:,}")
log_step_end("Step 5: Multi-Dimensional Aggregations", step_start)

# ============================================================================
# STEP 6: Cross-Tabulation and Pivot Operations
# ============================================================================
step_start = log_step_start("Step 6: Pivot Operations")

# Pivot survival by class and sex
pivot_survival = df_features.groupBy("PCLASS").pivot("SEX").agg(
    avg("SURVIVED").alias("survival_rate"),
    count("*").alias("count")
)

# Pivot fare statistics by embarkation and class
pivot_fare = df_features.groupBy("EMBARKED").pivot("PCLASS").agg(
    avg("FARE").alias("avg_fare"),
    max("FARE").alias("max_fare")
)

print(f"   Pivot operations completed")
log_step_end("Step 6: Pivot Operations", step_start)

# ============================================================================
# STEP 7: Self-Join for Comparative Analysis
# ============================================================================
step_start = log_step_start("Step 7: Self-Join Comparative Analysis")

# Compare passengers with similar characteristics
df_comparable = df_features.alias("p1").join(
    df_features.alias("p2"),
    (col("p1.PCLASS") == col("p2.PCLASS")) &
    (col("p1.SEX") == col("p2.SEX")) &
    (abs(col("p1.AGE") - col("p2.AGE")) < 5) &
    (col("p1.EMBARKED") == col("p2.EMBARKED")),
    "inner"
).select(
    col("p1.SURVIVED").alias("p1_survived"),
    col("p2.SURVIVED").alias("p2_survived"),
    col("p1.FARE").alias("p1_fare"),
    col("p2.FARE").alias("p2_fare"),
    col("p1.PCLASS").alias("pclass"),
    col("p1.SEX").alias("sex")
)

comparable_count = df_comparable.count()
print(f"   Comparable pairs found: {comparable_count:,}")
log_step_end("Step 7: Self-Join Comparative Analysis", step_start)

# ============================================================================
# STEP 8: Statistical Calculations - Correlation Matrix
# ============================================================================
step_start = log_step_start("Step 8: Statistical Correlations")

# Calculate correlations between numeric features
numeric_cols = ["SURVIVED", "PCLASS", "AGE", "SIBSP", "PARCH", "FARE", "family_size"]
correlations = []

for col1 in numeric_cols:
    for col2 in numeric_cols:
        if col1 != col2:
            corr = df_features.stat.corr(col1, col2)
            correlations.append((col1, col2, corr))
            
correlation_df = spark.createDataFrame(correlations, ["feature1", "feature2", "correlation"])
corr_count = correlation_df.count()
print(f"   Correlations computed: {corr_count}")
log_step_end("Step 8: Statistical Correlations", step_start)

# ============================================================================
# STEP 9: Complex UDF - Survival Score Calculation
# ============================================================================
step_start = log_step_start("Step 9: Custom UDF Application")

# Define complex survival prediction UDF
@udf(returnType=DoubleType())
def calculate_survival_score(pclass, sex, age, fare, family_size):
    """Complex scoring function simulating ML feature engineering"""
    score = 0.5  # Base score
    
    # Class impact
    if pclass == 1:
        score += 0.3
    elif pclass == 2:
        score += 0.1
    else:
        score -= 0.2
    
    # Sex impact
    if sex == "FEMALE":
        score += 0.4
    else:
        score -= 0.3
    
    # Age impact
    if age is not None:
        if age < 15:
            score += 0.2
        elif age > 60:
            score -= 0.1
    
    # Fare impact
    if fare is not None:
        if fare > 50:
            score += 0.15
        elif fare < 10:
            score -= 0.15
    
    # Family size impact
    if family_size is not None:
        if 2 <= family_size <= 4:
            score += 0.1
        elif family_size > 4:
            score -= 0.15
    
    # Clamp score between 0 and 1 using conditionals (PySpark UDF compatible)
    if score > 1:
        score = 1.0
    elif score < 0:
        score = 0.0
    
    return float(score)

df_scored = df_features.withColumn(
    "survival_score",
    calculate_survival_score(col("PCLASS"), col("SEX"), col("AGE"), col("FARE"), col("family_size"))
)

scored_count = df_scored.count()
print(f"   Survival scores calculated: {scored_count:,}")
log_step_end("Step 9: Custom UDF Application", step_start)

# ============================================================================
# STEP 10: Write Results to Multiple Iceberg Tables
# ============================================================================
step_start = log_step_start("Step 10: Write Results to Iceberg")

# Write main transformed dataset using SQL CREATE TABLE (AWS recommended for Glue 5.0)
print("   Writing titanic_transformed...")
df_scored.createOrReplaceTempView("tmp_titanic_transformed")
spark.sql("""
    CREATE OR REPLACE TABLE glue_catalog.tosmith_iceberg_project_db.titanic_transformed
    USING iceberg
    TBLPROPERTIES ('format-version'='2', 'write.parquet.compression-codec'='snappy')
    AS SELECT * FROM tmp_titanic_transformed
""")
print("   ✅ titanic_transformed written")

# Write aggregated statistics
print("   Writing survival_statistics...")
survival_stats.createOrReplaceTempView("tmp_survival_statistics")
spark.sql("""
    CREATE OR REPLACE TABLE glue_catalog.tosmith_iceberg_project_db.survival_statistics
    USING iceberg
    TBLPROPERTIES ('format-version'='2')
    AS SELECT * FROM tmp_survival_statistics
""")
print("   ✅ survival_statistics written")

# Write cohort analysis
print("   Writing cohort_analysis...")
cohort_analysis.createOrReplaceTempView("tmp_cohort_analysis")
spark.sql("""
    CREATE OR REPLACE TABLE glue_catalog.tosmith_iceberg_project_db.cohort_analysis
    USING iceberg
    TBLPROPERTIES ('format-version'='2')
    AS SELECT * FROM tmp_cohort_analysis
""")
print("   ✅ cohort_analysis written")

# Write family patterns
print("   Writing family_survival_patterns...")
family_patterns.createOrReplaceTempView("tmp_family_survival_patterns")
spark.sql("""
    CREATE OR REPLACE TABLE glue_catalog.tosmith_iceberg_project_db.family_survival_patterns
    USING iceberg
    TBLPROPERTIES ('format-version'='2')
    AS SELECT * FROM tmp_family_survival_patterns
""")
print("   ✅ family_survival_patterns written")
log_step_end("Step 10: Write Results to Iceberg", step_start)

# ============================================================================
# SUMMARY
# ============================================================================
job_end_time = time.time()
total_job_duration = job_end_time - job_start_time

print("\n" + "="*80)
print("TRANSFORMATION SUMMARY")
print("="*80)
print(f"Job completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total job duration: {total_job_duration:.2f}s ({total_job_duration/60:.2f} minutes)")
print("="*80)
print(f"\nOriginal rows processed: 891")
print(f"Features created: {len([c for c in df_scored.columns if c not in df.columns])}")
print(f"Aggregation tables created: 3")
print(f"Total output rows: {scored_count:,}")
print("\nTransformations applied:")
print("  ✅ Window functions (8 different windows)")
print("  ✅ Feature engineering (15+ new features)")
print("  ✅ Multi-dimensional aggregations")
print("  ✅ Pivot operations")
print("  ✅ Self-joins for comparative analysis")
print("  ✅ Statistical correlations")
print("  ✅ Custom UDFs")
print("  ✅ Multiple Iceberg table writes")

print("\n" + "="*80)
print("PERFORMANCE BREAKDOWN (by step)")
print("="*80)
for step_name, duration in step_timings:
    percentage = (duration / total_job_duration) * 100 if total_job_duration > 0 else 0
    print(f"{step_name:.<50} {duration:>8.2f}s ({percentage:>5.1f}%)")
print("="*80)
print(f"{'TOTAL':.<50} {total_job_duration:>8.2f}s (100.0%)")
print("="*80)

job.commit()