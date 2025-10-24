import os
import snowflake.snowpark
from snowflake import snowpark_connect
from pyspark.sql.functions import *
from pyspark.sql.types import *
# Import snowpark_connect before importing pyspark libraries
from pyspark.sql.types import Row

# Connection parameters - Update these with your Snowflake account details
os.environ["SPARK_CONNECT_MODE_ENABLED"] = "1"
snowpark_connect.start_session()  # Start the local Snowpark Connect for Spark session
spark = snowpark_connect.get_session()
spark.conf.set("snowpark.connect.sql.identifiers.auto-uppercase", "none")
# spark.catalog.setCurrentCatalog('aws_glue_catalog_linked_db')
# spark.catalog.setCurrentDatabase('tosmith_iceberg_project_db')

df = spark.table('AWS_GLUE_CATALOG_LINKED_DB.tosmith_iceberg_project_db.titanic_transformed')

# df = spark.sql('SELECT * FROM `titanic_transformed`')

print(df.count())