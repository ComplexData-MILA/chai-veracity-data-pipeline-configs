"""Validate classifier features

(test data split.)
"""

import asyncio
import json

from s3_data_tool import S3DataTool, RawDuckFilter


async def main():
    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name="dry_run_data_export_001",
        base_columns=["id", "text"],
        annotator_columns={"feasibility_llm_judge_001": ["is_feasible"]},
        annotator_filters={
            "feasibility_llm_judge_001": RawDuckFilter(sql="is_feasible IS NOT NULL"),
        },
    ) as annotator_view:
        async for _row in annotator_view:
            print(json.dumps(_row.data, indent=2))
# {
#   "id": "afcdc7...",
#   "_batch": "batch_name",
#   "text": "Example text- might contain unicode",
#   "is_feasible": 0
# }


if __name__ == "__main__":
    asyncio.run(main())
