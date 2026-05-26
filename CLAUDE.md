Use secrets from .env to access the data.

Documentation for the data interface library (S3 Data Tool) are available within [this repository](https://github.com/complexData-MILA/data-lake-pipeline). You may make a copy of that under /tmp for analysis as needed. The data is stored in S3 as parquet files, and can be accessed directly for debug purposes. 

When asked to monitor a long-running job, use the job-monitor agent as a pre-filter for the events that are generated (so as to save your own context.)