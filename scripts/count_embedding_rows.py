"""Count rows needing embedding annotation.

Replicates the query logic from FilterForAnnotation._iter_filtered_items
but wraps it in SELECT COUNT(*) to get an exact row count.
"""

import asyncio
import os
import tempfile
import shutil

import aioboto3
import duckdb

from s3_data_tool.s3_utils import enumerate_parquet_paths
from s3_data_tool.filter import RawDuckFilter
from s3_data_tool.data_filtering import FilterForExport


async def main():
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    access_key = os.environ["S3_ACCESS_KEY"]
    secret_key = os.environ["S3_SECRET_KEY"]

    dataset_name = "posts"
    annotator_name = "embeddings_128d"

    session = aioboto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    async with session.client("s3", **kwargs) as s3_client:

        # Build a minimal FilterForExport to reuse _build_filtered_cte
        ffe = FilterForExport(
            s3_client=s3_client,
            bucket=bucket,
            prefix=prefix,
            dataset_name=dataset_name,
            base_columns=["text"],
            annotator_columns={"feasibility_classifier_003": ["classifier_label"]},
            annotator_filters={
                "feasibility_classifier_003": RawDuckFilter(
                    sql="classifier_label = '\"LABEL_2\"'"
                ),
            },
        )

        query = await ffe.get_duckdb_query()
        # get_duckdb_query doesn't handle the annotator_filters param properly
        # (it uses self._annotator_filters but get_duckdb_query on FilterForExport
        # does use annotator_filters in _build_filtered_cte). Let me verify by
        # reconstructing the query manually since we need anti-join too.

        # Path enumeration
        base_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, dataset_name
        )
        annotator_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, dataset_name,
            annotator_name,
        )
        feasibility_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, dataset_name,
            "feasibility_classifier_003",
        )

    # Build query manually in sync DuckDB (static data, no async needed)
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "count.duckdb")
    conn = duckdb.connect(db_path, read_only=False)

    try:
        endpoint_host = endpoint_url.removeprefix("https://").rstrip("/")
        endpoint_host = endpoint_host.removeprefix("http://").rstrip("/")
        use_ssl = endpoint_url.startswith("https://")

        conn.execute(f"""
            SET s3_access_key_id='{access_key}';
            SET s3_secret_access_key='{secret_key}';
            SET s3_endpoint='{endpoint_host}';
            SET s3_use_ssl={str(use_ssl).lower()};
            SET s3_url_style='path';
        """)

        # Build CTEs manually matching _iter_filtered_items logic
        def format_paths(paths):
            return ", ".join(f"'{p}'" for p in paths)

        # Base CTE with no filter
        base_cte = (
            f"base_filtered AS ("
            f"SELECT id, ANY_VALUE(_batch) AS _batch, ANY_VALUE(text) AS text "
            f"FROM read_parquet([{format_paths(base_paths)}]) "
            f"GROUP BY id)"
        )

        # feasibility_classifier_003 CTE with filter
        feasibility_cte = (
            f"feasibility_classifier_003_filtered AS ("
            f"SELECT id, ANY_VALUE(classifier_label) AS classifier_label "
            f"FROM read_parquet([{format_paths(feasibility_paths)}]) "
            f"WHERE classifier_label = '\"LABEL_2\"' "
            f"GROUP BY id)"
        )

        # annotator_done CTE (anti-join)
        annotator_done_cte = ""
        anti_join = ""
        if annotator_paths:
            annotator_done_cte = (
                f"annotator_done AS ("
                f"SELECT id FROM read_parquet([{format_paths(annotator_paths)}]))"
            )
            anti_join = (
                " LEFT JOIN annotator_done USING (id) "
                "WHERE annotator_done.id IS NULL"
            )

        ctes = [base_cte, feasibility_cte]
        if annotator_done_cte:
            ctes.append(annotator_done_cte)

        query = (
            f"WITH {', '.join(ctes)} "
            f"SELECT COUNT(*) AS cnt "
            f"FROM base_filtered "
            f"JOIN feasibility_classifier_003_filtered USING (id)"
            f"{anti_join}"
        )

        print(f"Running count query...")
        print(f"Base paths: {len(base_paths)}")
        print(f"Feasibility paths: {len(feasibility_paths)}")
        print(f"Existing embedding paths: {len(annotator_paths)}")

        result = conn.execute(query)
        count = result.fetchone()[0]
        print(f"\n=== Rows needing embedding annotation: {count:,} ===")

        # Also count total per batch for breakdown
        batch_query = (
            f"WITH {', '.join(ctes)} "
            f"SELECT _batch, COUNT(*) AS cnt "
            f"FROM base_filtered "
            f"JOIN feasibility_classifier_003_filtered USING (id)"
            f"{anti_join} "
            f"GROUP BY _batch ORDER BY cnt DESC"
        )
        print("\n--- Per-batch breakdown ---")
        for row in conn.execute(batch_query).fetchall():
            print(f"  {row[0]}: {row[1]:,}")

    finally:
        conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
