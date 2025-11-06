import io
import logging
from typing import Generator, List, Optional

import boto3
import botocore
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Stream CC100 Arrow files directly from S3 with no local disk usage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Path/prefix in the bucket (e.g. "cc100").
    language : str, default=""
        Optional filter on filenames.
    max_files : int, optional
        Max number of shards to read.
    batch_size : int, default=20000
        Size of yielded mini-batches.
    max_phrases : int, optional
        Global cap on number of phrases to load from S3.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        language: str = "",
        max_files: Optional[int] = None,
        batch_size: int = 20_000,
        max_phrases: Optional[int] = None,  # ✅ added
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.language = language
        self.max_files = max_files
        self.batch_size = batch_size
        self.max_phrases = max_phrases  # ✅ added

        # Explicit region
        session = boto3.session.Session()
        region = session.region_name or "us-east-1"
        self.s3 = boto3.client("s3", region_name=region)

        logger.info(f"Scanning S3: s3://{bucket}/{self.prefix}")

        self.files = self._list_s3_shards()

        if not self.files:
            raise FileNotFoundError(
                f"No Arrow/Parquet shards under s3://{bucket}/{self.prefix}"
            )

        logger.info(f"Found {len(self.files)} S3 shards.")

        if max_files is not None:
            self.files = self.files[:max_files]
            logger.info(f"Limiting to first {max_files} shards.")

    # ------------------------------------------------------------------
    def _list_s3_shards(self) -> List[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        keys: List[str] = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".arrow") or key.endswith(".parquet"):
                    if not self.language or self.language.lower() in key.lower():
                        keys.append(key)

        return sorted(keys)

    # ------------------------------------------------------------------
    def _load_arrow_from_s3(self, key: str) -> pa.Table:
        logger.info(f"Downloading: s3://{self.bucket}/{key}")

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(
                f"Failed to download s3://{self.bucket}/{key}"
            ) from e

        raw_bytes = resp["Body"].read()
        buf = io.BytesIO(raw_bytes)

        # Try arrow file first
        try:
            with pa_ipc.open_file(buf) as reader:
                return reader.read_all()
        except pa.lib.ArrowInvalid:
            buf.seek(0)
            with pa_ipc.open_stream(buf) as reader:
                return reader.read_all()

    # ------------------------------------------------------------------
    def _table_to_df(self, table: pa.Table) -> pd.DataFrame:
        df = table.to_pandas()
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})
        return df.dropna(subset=["phrase"])

    # ------------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Stream rows from S3 in mini-batches, respecting max_phrases.
        """
        yielded_total = 0

        for shard_idx, key in enumerate(self.files, start=1):
            logger.info(f"[{shard_idx}/{len(self.files)}] Loading shard: {key}")

            table = self._load_arrow_from_s3(key)
            df = self._table_to_df(table)

            total_rows = len(df)
            logger.info(f"Shard rows: {total_rows:,}")

            # yield in batches
            for start in range(0, total_rows, self.batch_size):
                sub = df.iloc[start : start + self.batch_size]
                batch_len = len(sub)

                # handle max_phrases cutoff
                if self.max_phrases is not None:
                    remaining = self.max_phrases - yielded_total
                    if remaining <= 0:
                        logger.info("Reached max_phrases limit; stopping stream.")
                        return
                    if batch_len > remaining:
                        sub = sub.iloc[:remaining]
                        logger.info("Trimmed final batch due to max_phrases.")

                yielded_total += len(sub)

                logger.info(
                    f"Yielding {len(sub):,} rows "
                    f"({start // self.batch_size + 1}/"
                    f"{(total_rows + self.batch_size - 1) // self.batch_size})"
                )

                yield sub

                # stop early if limit reached
                if self.max_phrases is not None and yielded_total >= self.max_phrases:
                    logger.info("Reached max_phrases limit; stopping stream.")
                    return
