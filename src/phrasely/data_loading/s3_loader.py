import logging
from typing import Generator, Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Streams CC100 Arrow/Parquet shards directly from S3 *without* saving to disk.

    This mirrors the behavior of CC100OfflineLoader but reads files via boto3
    using in-memory Arrow buffers.

    Parameters
    ----------
    bucket : str
        S3 bucket name, e.g. "phrasely-data-mastroianni".
    prefix : str
        Prefix under bucket where shards reside, e.g. "cc100".
    language : str, default="en"
        Optional filter applied to filenames. Set to "" to load all.
    max_files : int, optional
        Cap the number of files (useful for debugging).
    batch_size : int, default=20_000
        Rows returned per mini-batch.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        language: str = "en",
        max_files: Optional[int] = None,
        batch_size: int = 20_000,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.language = language
        self.max_files = max_files
        self.batch_size = batch_size

        self.s3 = boto3.client("s3")

        logger.info(f"Scanning S3: s3://{bucket}/{prefix}")

        # -------------------------------
        # ✅ FIXED: Proper S3 pagination
        # -------------------------------
        all_files = []
        continuation = None

        while True:
            if continuation:
                resp = self.s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{self.prefix}/",
                    ContinuationToken=continuation,
                )
            else:
                resp = self.s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{self.prefix}/",
                )

            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".arrow") or key.endswith(".parquet"):
                    if not language or language.lower() in key.lower():
                        all_files.append(key)

            if resp.get("IsTruncated"):
                continuation = resp["NextContinuationToken"]
            else:
                break

        if not all_files:
            raise FileNotFoundError(
                f"No Arrow/Parquet files found under s3://{bucket}/{prefix}"
            )

        if max_files is not None:
            logger.warning(
                f"Found {len(all_files)} shards; limiting to first {max_files}"
            )
            all_files = all_files[:max_files]

        self.files = sorted(all_files)

        logger.info(f"✅ Found {len(self.files)} S3 shards.")

    # ----------------------------------------------------------------------
    def _load_arrow_from_s3(self, key: str) -> pa.Table:
        """Download an S3 Arrow/Parquet file into a PyArrow Table."""
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"].read()

        # Detect Arrow vs Parquet
        if key.endswith(".parquet"):
            return pq.read_table(pa.BufferReader(body))

        # Arrow: try file format, fallback to stream
        buf = pa.BufferReader(body)
        try:
            reader = pa_ipc.open_file(buf)
            return reader.read_all()
        except pa.lib.ArrowInvalid:
            reader = pa_ipc.open_stream(buf)
            return reader.read_all()

    # ----------------------------------------------------------------------
    def _table_to_df(self, table: pa.Table) -> pd.DataFrame:
        df = table.to_pandas()

        # Normalize column naming
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})

        df = df.dropna(subset=["phrase"])
        return df

    # ----------------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Yield DataFrame mini-batches sequentially across all shards.
        """
        logger.info(f"📥 Streaming from {len(self.files)} files in S3…")

        for key in self.files:
            logger.info(f"→ Loading: {key}")

            table = self._load_arrow_from_s3(
