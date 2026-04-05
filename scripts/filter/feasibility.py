import asyncio

from s3_data_tool import Annotation, DataItem, S3DataTool

streaming_configs = S3DataTool.StreamingConfigs(chunk_size=10)


async def annotate(item: DataItem) -> Annotation:
    return Annotation(
        data={"is_feasible": True, "explanation": f"Source: {item.data["text"]}"}
    )


async def main():
    # If another annotation worker with the same annotator_name is active and not expired,
    # the following should raise an Error.
    async with S3DataTool().filter_for_annotation(
        name="posts",
        annotator_name="custom_search",
        base_columns=["text"],
    ) as annotator_view:
        await annotator_view.annotate(
            annotate,
            max_concurrency=16,  # concurrency limits for the "annotate" function
            streaming_configs=streaming_configs,
        )


if __name__ == "__main__":
    asyncio.run(main())
