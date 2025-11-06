import io
import logging
from typing import Generator, List, Optional

import boto3
import botocore
import pyarrow as pa
import pyarrow.ipc as pa_ipc

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Stream CC100 Arrow files directly from S3 without local storage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        The directory/prefix inside the bucket (e.g. "cc100").
    language : str, default="en"
        Language filter for filenames.
    max_files : int, optional
        Maximum number of shards to read.
    batch_size : int, default=20000
        Rows per yielded batch.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        language: str = "",
        max_files: Optional[int] = None,
        batch_size: int = 20_000,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")  # ensure no trailing slash
        self.language = language
        self.max_files = max_files
        self.batch_size = batch_size

        # ✅ Explicit region (us-east-1 bucket) – fixes empty list_objects issue
        session = boto3.session.Session()
        region = session.region_name or "us-east-1"

        self.s3 = boto3.client("s3", region_name=region)

        logger.info(f"Scanning S3: s3://{bucket}/{self.prefix}")

        self.files = self._list_s3_shards()
        if not self.files:
            raise FileNotFoundError(
                f"No Arrow/Parquet files found under s3://{bucket}/{self.prefix}"
            )

        logger.info(f"Found {len(self.files)} S3 shards.")

        if max_files is not None:
            self.files = self.files[:max_files]
            logger.info(f"Limiting to first {max_files} shards.")

    # ------------------------------------------------------------------
    def _list_s3_shards(self) -> List[str]:
        """List Arrow/Parquet files under prefix."""
        paginator = self.s3.get_paginator("list_objects_v2")

        keys: List[str] = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".arrow") or key.endswith(".parquet"):
                    if not self.language or self.language in key.lower():
                        keys.append(key)

        return sorted(keys)

    # ------------------------------------------------------------------
    def _load_arrow_from_s3(self, key: str) -> pa.Table:
        """Download arrow/parquet shard from S3 into memory and return Arrow table."""
        logger.info(f"Downloading: s3://{self.bucket}/{key}")

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(f"Failed to download s3://{self.bucket}/{key}") from e

        raw_bytes = resp["Body"].read()
        buf = io.BytesIO(raw_bytes)

        # First try Arrow file reader
        try:
            with pa_ipc.open_file(buf) as reader:
                return reader.read_all()
        except pa.lib.ArrowInvalid:
            buf.seek(0)
            with pa_ipc.open_stream(buf) as reader:
                return reader.read_all()

    # ------------------------------------------------------------------
    def _table_to_df(self, table: pa.Table):
        """Normalize Arrow table into DataFrame with 'phrase' column."""
        df = table.to_pandas()

        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})

        df = df.dropna(subset=["phrase"])
        return df

    # ------------------------------------------------------------------
    def stream_load(self) -> Generator:
        """
        Yield batches from S3 shards as Pandas DataFrames.

        Fully streaming — no local disk usage.
        """

        for idx, key in enumerate(self.files, start=1):
            logger.info(f"[{idx}/{len(self.files)}] Loading shard: {key}")

            table = self._load_arrow_from_s3(key)
            df = self._table_to_df(table)

            total_rows = len(df)
            logger.info(f"Shard rows: {total_rows:,}")

            # yield small DataFrames
            for start in range(0, total_rows, self.batch_size):
                sub = df.iloc[start : start + self.batch_size]

                logger.info(
                    f"Yielding {len(sub):,} rows "
                    f"({start // self.batch_size + 1}/"
                    f"{(total_rows + self.batch_size - 1) // self.batch_size})"
                )
                yield sub
