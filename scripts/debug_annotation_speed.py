"""Debug script: time each step of the annotation pipeline for a single batch."""

import asyncio
import os
import time

from s3_data_tool import RawDuckFilter, S3DataTool
from s3_data_tool.data_filtering import FilterForAnnotation
import aioboto3


async def main():
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "datasets")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    access_key = os.environ["S3_ACCESS_KEY"]
    secret_key = os.environ["S3_SECRET_KEY"]

    dataset_name = "posts"
    annotator_name = "embeddings_128d_dryrun"

    session = aioboto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    async with session.client("s3", **kwargs) as s3_client:
        # Step 1: List base paths
        from s3_data_tool.s3_utils import enumerate_parquet_paths

        t0 = time.monotonic()
        base_paths = await enumerate_parquet_paths(s3_client, bucket, prefix, dataset_name)
        t1 = time.monotonic()
        print(f"Step 1 - enumerate base paths ({len(base_paths)} paths): {t1-t0:.1f}s")

        # Step 2: List feasibility paths
        feasibility_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, dataset_name, "feasibility_classifier_003"
        )
        t2 = time.monotonic()
        print(f"Step 2 - enumerate feasibility paths ({len(feasibility_paths)} paths): {t2-t1:.1f}s")

        # Step 3: List existing annotator paths (should be empty for dryrun)
        annotator_paths = await enumerate_parquet_paths(
            s3_client, bucket, prefix, dataset_name, annotator_name
        )
        t3 = time.monotonic()
        print(f"Step 3 - enumerate annotator paths ({len(annotator_paths)} paths): {t3-t2:.1f}s")

        # Step 4: Pick a single batch and time DuckDB query
        test_batch = base_paths[0].split("/")[-2]
        batch_paths = [p for p in base_paths if f"/{test_batch}/" in p]
        batch_feasibility = [p for p in feasibility_paths if f"/{test_batch}/" in p]

        print(f"\nTesting batch: {test_batch}")
        print(f"  Base paths: {len(batch_paths)}")
        print(f"  Feasibility paths: {len(batch_feasibility)}")

        # Build query manually (same as _iter_filtered_items)
        def format_paths(paths):
            return ", ".join(f"'{p}'" for p in paths)

        import tempfile, shutil
        import duckdb

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "debug.duckdb")
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

            # COUNT query
            base_cte = (
                f"base_filtered AS ("
                f"SELECT id, ANY_VALUE(_batch) AS _batch, ANY_VALUE(text) AS text "
                f"FROM read_parquet([{format_paths(batch_paths)}]) "
                f"GROUP BY id)"
            )
            feasibility_cte = (
                f"feasibility_classifier_003_filtered AS ("
                f"SELECT id, ANY_VALUE(classifier_label) AS classifier_label "
                f"FROM read_parquet([{format_paths(batch_feasibility)}]) "
                f"WHERE classifier_label = '\"LABEL_2\"' "
                f"GROUP BY id)"
            )

            # Count total after join (no anti-join, no sampling)
            count_query = (
                f"WITH {base_cte}, {feasibility_cte} "
                f"SELECT COUNT(*) AS cnt "
                f"FROM base_filtered "
                f"JOIN feasibility_classifier_003_filtered USING (id)"
            )
            t4 = time.monotonic()
            result = conn.execute(count_query)
            count = result.fetchone()[0]
            t5 = time.monotonic()
            print(f"\nStep 4 - COUNT query: {count:,} rows, took {t5-t4:.1f}s")

            # Step 5: Test SELECT with streaming (fetch first 100 rows)
            select_query = (
                f"WITH {base_cte}, {feasibility_cte} "
                f"SELECT base_filtered.*, feasibility_classifier_003_filtered.* "
                f"FROM base_filtered "
                f"JOIN feasibility_classifier_003_filtered USING (id)"
            )
            t6 = time.monotonic()
            result = conn.execute(select_query)
            # Use fetchmany with LIMIT to simulate streaming
            desc = result.description
            rows = result.fetchmany(100)
            t7 = time.monotonic()
            print(f"Step 5 - SELECT first 100 rows: {len(rows)} rows, took {t7-t6:.1f}s")
            if rows:
                print(f"  First row keys: {list({desc[i][0]: rows[0][i] for i in range(len(desc))}.keys())[:5]}")

        finally:
            conn.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\nTotal time: {t7-t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
