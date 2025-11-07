import io
import logging
from typing import List, Optional

import boto3
import botocore
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Streams CC100 Arrow/Parquet shards directly from S3 with zero local disk usage.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Subdirectory or prefix within the bucket.
    language : str, default=""
        Optional case-insensitive substring filter for filenames.
    max_files : int, optional
        Cap the number of shards to consider.
    batch_size : int, default=20000
        Number of phrases per yielded DataFrame batch.
    max_phrases : int, optional
        Global cap on total phrases yielded across all shards.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        language: str = "",
        max_files: Optional[int] = None,
        batch_size: int = 20_000,
        max_phrases: Optional[int] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.language = language.lower() if language else ""
        self.max_files = max_files
        self.batch_size = batch_size
        self.max_phrases = max_phrases

        # Pick an explicit region (avoids None-region issues in boto3)
        session = boto3.session.Session()
        region = session.region_name or "us-east-1"

        # boto3 client
        self.s3 = boto3.client("s3", region_name=region)

        logger.info(f"Scanning S3 prefix: s3://{bucket}/{self.prefix}/")

        # List all shards
        self.files = self._list_s3_shards()
        if not self.files:
            raise FileNotFoundError(f"No Arrow/Parquet shards under s3://{bucket}/{self.prefix}/")

        logger.info(f"Discovered {len(self.files)} shards.")

        # Optional cap
        if max_files is not None:
            self.files = self.files[:max_files]
            logger.info(f"Restricted to first {max_files} shards.")

    # ------------------------------------------------------------------
    def _list_s3_shards(self) -> List[str]:
        """
        List all .arrow or .parquet shards under the given bucket/prefix.
        Language filters are applied here to avoid unnecessary downloads.
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        keys: List[str] = []
        prefix = f"{self.prefix}/"

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith((".arrow", ".parquet")):
                    if not self.language or self.language in key.lower():
                        keys.append(key)

        return sorted(keys)

    # ------------------------------------------------------------------
    def _load_arrow_from_s3(self, key: str) -> pa.Table:
        """
        Download a shard and decode it as either a file or a stream.
        """
        uri = f"s3://{self.bucket}/{key}"
        logger.info(f"Downloading shard: {uri}")

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            raise FileNotFoundError(f"Could not download {uri}") from e

        raw_bytes = resp["Body"].read()
        buf = io.BytesIO(raw_bytes)

        # Try Arrow file, then Arrow stream
        try:
            with pa_ipc.open_file(buf) as reader:
                table = reader.read_all()
        except pa.lib.ArrowInvalid:
            buf.seek(0)
            with pa_ipc.open_stream(buf) as reader:
                table = reader.read_all()

        return table

    # ------------------------------------------------------------------
    @staticmethod
    def _table_to_df(table: pa.Table) -> pd.DataFrame:
        """
        Convert pyarrow.Table to pandas.DataFrame and ensure a 'phrase' column.
        """
        df = table.to_pandas()

        # Normalize columns: text → phrase
        if "text" in df.columns and "phrase" not in df.columns:
            df = df.rename(columns={"text": "phrase"})

        if "phrase" not in df.columns:
            raise ValueError("CC100 shard is missing a 'phrase' or 'text' column.")

        # Drop missing rows
        df = df.dropna(subset=["phrase"])
        return df

    # ------------------------------------------------------------------
    def stream_load(self):
        """
        Stream phrases in mini-batches as pandas DataFrames.
        Respects `max_phrases` and `batch_size`.
        """
        yielded_total = 0

        for idx, key in enumerate(self.files, start=1):
            logger.info(f"[{idx}/{len(self.files)}] Loading shard {key}")

            table = self._load_arrow_from_s3(key)
            df = self._table_to_df(table)
            total_rows = len(df)

            logger.info(f"Shard rows: {total_rows:,}")

            # Yield in sub-batches
            for start in range(0, total_rows, self.batch_size):
                sub = df.iloc[start : start + self.batch_size]
                batch_len = len(sub)

                # Respect global phrase cap
                if self.max_phrases is not None:
                    remaining = self.max_phrases - yielded_total
                    if remaining <= 0:
                        logger.info("Reached max_phrases limit.")
                        return
                    if batch_len > remaining:
                        sub = sub.iloc[:remaining]

                yielded_total += len(sub)

                logger.info(
                    f"Yielding {len(sub):,} rows "
                    f"({start // self.batch_size + 1}/"
                    f"{(total_rows + self.batch_size - 1) // self.batch_size})"
                )

                yield sub

                # Early stop if global cap reached
                if self.max_phrases is not None and yielded_total >= self.max_phrases:
                    logger.info("Reached max_phrases limit.")
                    return
