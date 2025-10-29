# Snowpark Connect for Spark on Glue-managed Iceberg

## TL;DR
### AWS Glue running Spark: 30 minutes. 10 G.2x workers

### Snowflake running same code: 4.5 minutes. 1 Medium WH

## Intro
This project demonstrates running Apache Spark workloads on Snowflake using **Snowpark Connect for Spark** with **Apache Iceberg™** tables managed by AWS Glue Catalog.

## Overview

This repository contains scripts that showcase enterprise-grade Spark patterns (window functions, feature engineering, aggregations, UDFs) running on Snowflake's execution engine while reading and writing data to external managed Iceberg tables in AWS S3 via AWS Glue Catalog.

### Key Components

- **`tpch_snowpark_connect_script.py`**: Main transformation script demonstrating complex Spark operations on Snowflake
- **`tpch_glue_script.py`**: Original TPCH-100 AWS Glue script (for comparison)
- **`titanic_snowpark_connect_script.py`**: Main transformation script demonstrating complex Spark operations on Snowflake
- **`snowpark_connect_test.py`**: Simple connectivity test script for Snowpark Connect
- **`titanic_snowflake.csv`**: Sample Titanic dataset
- **`titanic_glue_script.py`**: Original titanic AWS Glue script (for comparison)


## Prerequisites

- Python 3.8+
- Snowflake account with Snowpark Connect for Spark enabled
- AWS Glue Catalog with Iceberg tables
- S3 bucket with Iceberg data
- Snowflake catalog integration linked to AWS Glue Catalog

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd aws-glue
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
```
snowflake-dataframe-processor==0.22.1
snowflake-snowpark-python
pyspark
```

## Configuration

### Snowpark Connect Setup

The scripts use Snowpark Connect for Spark, which requires the following setup:

```python
import os
from snowflake import snowpark_connect

# Enable Spark Connect mode
os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"

# Start Snowpark Connect session
snowpark_connect.start_session()
spark = snowpark_connect.get_session()

# Configure identifier handling
spark.conf.set("snowpark.connect.sql.identifiers.auto-uppercase", "none")
```

### AWS Glue Catalog Integration

This project uses an external managed Iceberg table configuration where:
- **Catalog**: AWS Glue Catalog linked to Snowflake
- **Database**: `tosmith_iceberg_project_db`
- **Tables**: Stored in S3 as Iceberg format
- **Access**: Through Snowflake catalog integration named `AWS_GLUE_CATALOG_LINKED_DB`

Example table reference:
```python
df = spark.table('AWS_GLUE_CATALOG_LINKED_DB.tosmith_iceberg_project_db.titanic_test')
```

## Usage

### Quick Test

Run the connectivity test to verify your setup:

```bash
python snowpark_connect_test.py
```

### Full Transformation Pipeline

Execute the main transformation script:

```bash
python titanic_snowpark_connect_script.py
```

This script performs:
1. **Data Ingestion**: Reads Iceberg tables from AWS Glue Catalog
2. **Window Functions**: Calculates rankings, percentiles, and moving averages
3. **Feature Engineering**: Creates derived features and interactions
4. **Aggregations**: Multi-dimensional survival statistics
5. **Pivot Operations**: Cross-tabulated survival analysis
6. **Self-Joins**: Comparative passenger analysis
7. **Statistical Analysis**: Correlation matrices
8. **Custom UDFs**: Survival score calculation
9. **Data Output**: Writes results to multiple Iceberg tables

## Snowpark Connect for Spark - Iceberg Compatibility

> **Note**: This information is based on the [Snowflake Snowpark Connect for Spark compatibility guide](https://docs.snowflake.com/en/developer-guide/snowpark-connect/snowpark-connect-compatibility).

### External Managed Iceberg Table Support

When working with external managed Iceberg tables (tables managed outside of Snowflake, such as those in AWS Glue Catalog), be aware of the following compatibility considerations:

#### Read Operations

✅ **Supported**:
- Reading from external managed Iceberg tables
- Standard DataFrame operations on Iceberg data

❌ **Not Supported**:
- Time travel (historical snapshots)
- Branch reads
- Incremental reads

⚠️ **Requirements**:
- You **must** create a Snowflake unmanaged table entity that points to the external Iceberg table
- This is typically done through a Snowflake catalog integration

#### Write Operations

✅ **Supported**:
- Writing to existing Iceberg tables

❌ **Not Supported**:
- Table creation through Spark SQL (e.g., `CREATE TABLE ... USING iceberg`)

### Snowflake Managed Iceberg Table Support

For Snowflake-managed Iceberg tables, there are additional requirements:

#### Write Requirements

To create tables, you must:
1. Create an external volume in Snowflake
2. Link the external volume to table creation by either:
   - Setting `EXTERNAL_VOLUME` on the database
   - Setting `snowpark.connect.iceberg.external_volume` in Spark configuration

#### Limitations

❌ **Not Supported**:
- Time travel operations
- Schema merge operations
- Using Spark SQL to create tables

### Data Source Compatibility

When working with file formats and Snowflake tables:

| Feature | Status | Notes |
|---------|--------|-------|
| Parquet read/write | ⚠️ Partial | Append/Ignore modes not supported; compression options limited |
| CSV read/write | ⚠️ Partial | Many encoding and parsing options not supported |
| JSON read/write | ⚠️ Partial | Limited option support; display formatting differences |
| ORC | ❌ Not supported | - |
| Avro | ❌ Not supported | - |
| Snowflake tables | ✅ Supported | No provider format needed; partitioning/bucketing not supported |

### Catalog API Support

When using `spark.catalog` APIs:

✅ **Fully Supported**:
- `listDatabases()`
- `listTables()`
- `tableExists()`
- `currentDatabase()`

❌ **Not Supported**:
- `registerFunction()`
- `listFunctions()`
- `getFunction()`
- `createExternalTable()`

⚠️ **Partially Supported**:
- `createTable()` - No external table support

### Important Data Type Considerations

Snowpark Connect for Spark implicitly converts certain data types:

| Original Spark Type | Snowpark Connect Type |
|---------------------|----------------------|
| `ByteType` | `LongType` |
| `ShortType` | `LongType` |
| `IntegerType` | `LongType` |
| `FloatType` | `DoubleType` (context-dependent) |

⚠️ **Note**: This won't affect query correctness but may appear in schema outputs.

### UDF Differences

When using User-Defined Functions:

**StructType Handling**:
- **Native Spark**: Converts to Python `tuple` (access via `e[0], e[1]`)
- **Snowpark Connect**: Converts to Python `dict` (access via `e['_1'], e['_2']`)

**Example workaround**:
```python
# Instead of:
def f(e):
    return e[0]

# Use:
def f(e):
    return e['_1']
```

❌ **Not Supported**:
- Iterator types in UDF parameters or return values
- UDFs within lambda expressions

### Lambda Function Limitations

❌ **Not Supported**:
- Referencing outer columns/expressions from within lambda body
- UDFs inside lambda expressions

```python
# ❌ This will fail:
df.select(transform(df.numbers, lambda el: el + array_size(df.numbers)))

# ✅ This works:
df.select(transform(df.numbers, lambda el: negative(el) + 1))
```

### Duplicate Column Names

Snowflake does not support duplicate column names. To handle cases where Spark operations might create duplicate columns:

**Configuration option**:
```python
spark.conf.set("snowpark.connect.views.duplicate_column_names_handling_mode", "rename")
```

Options:
- `rename`: Appends `_dedup_1`, `_dedup_2`, etc. to duplicates
- `drop`: Drops all duplicate columns except one (may cause data loss)

### Unsupported Spark APIs

The following common Spark APIs are not supported or are no-ops:

- `DataFrame.hint()` - Ignored (Snowflake optimizer handles this)
- `DataFrame.repartition()` - No-op (Snowflake handles partitioning automatically)
- `pyspark.RDD` - RDD APIs not supported
- `pyspark.ml` - MLlib not supported
- `pyspark.streaming` - Streaming not supported

## Project Structure

```
aws-glue/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── snowpark_connect_test.py              # Simple connectivity test
├── titanic_snowpark_connect_script.py    # Main transformation script
├── titanic_glue_script.py                # Original AWS Glue version
├── titanic_snowflake.csv                 # Sample dataset
├── activate_venv.txt                     # Virtual environment notes
└── pmpt.txt                              # Additional notes
```

## Output Tables

The transformation pipeline creates the following Iceberg tables:

1. **`titanic_transformed`**: Main dataset with all engineered features and scores
2. **`survival_statistics`**: Multi-dimensional survival analysis by class, sex, age, and fare buckets
3. **`cohort_analysis`**: Survival patterns by embarkation port and passenger class
4. **`family_survival_patterns`**: Survival rates by family size and class

## Performance Considerations

- **Window Functions**: Most compute-intensive operations (typically 40-60% of runtime)
- **Cross Joins**: Creates large intermediate datasets (891 × 891 = 793,881 rows)
- **UDFs**: Custom Python UDFs may have performance overhead
- **Iceberg Writes**: Writing to multiple tables in sequence

The script includes detailed timing instrumentation for each step.

## Key Differences from AWS Glue

| Aspect | AWS Glue | Snowpark Connect for Spark |
|--------|----------|---------------------------|
| Context | `GlueContext` | Standard `SparkSession` |
| Catalog Read | `create_data_frame.from_catalog()` | `spark.table()` |
| Catalog | AWS Glue Catalog | Snowflake + Glue Catalog Integration |
| Execution | AWS Glue (Spark on EMR) | Snowflake compute resources |
| Table Prefix | `glue_catalog.db.table` | `CATALOG_INTEGRATION.db.table` |

## Troubleshooting

### Common Issues

1. **Table not found**: Ensure AWS Glue Catalog integration is properly configured in Snowflake
2. **Permission errors**: Verify IAM roles and Snowflake external integration permissions
3. **Data type errors**: Check that numeric columns are properly cast (see script for examples)
4. **Write failures**: Confirm external volume is configured for Iceberg writes

### Debug Commands

```python
# List available databases
spark.catalog.listDatabases()

# List tables in a database
spark.sql("SHOW TABLES IN AWS_GLUE_CATALOG_LINKED_DB.tosmith_iceberg_project_db").show()

# Check Spark configuration
spark.sparkContext.getConf().getAll()
```

## Additional Resources

- [Snowpark Connect for Spark Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-connect)
- [Snowpark Connect Compatibility Guide](https://docs.snowflake.com/en/developer-guide/snowpark-connect/snowpark-connect-compatibility)
- [Iceberg Table Support](https://docs.snowflake.com/en/developer-guide/snowpark-connect/snowpark-connect-compatibility#external-managed-iceberg-table)
- [Apache Iceberg Documentation](https://iceberg.apache.org/)
- [AWS Glue Data Catalog](https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html)

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]

---

**Last Updated**: October 2025

For questions or issues, please contact the project maintainer.

