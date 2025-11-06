import logging
from typing import Generator, Optional

import botocore
import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Stream CC100 Arrow/Parquet shards directly from S3 without disk writes.
    Matches CC100OfflineLoader semantics.
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

        self.s3 = boto3.client("s3", region_name="us-east-1")

        logger.info(f"Scanning S3: s3://{bucket}/{prefix}")

        # ---- List all S3 objects under prefix (paginated) ----
        all_files = []
        continuation = None

        while True:
            kwargs = {
                "Bucket": bucket,
                "Prefix": f"{self.prefix}/",
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation

            resp = self.s3.list_objects_v2(**kwargs)

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
            all_files = all_files[:max_files]

        self.files = sorted(all_files)
        logger.info(f"✅ Found {len(self.files)} matching S3 shards.")

    # ------------------------------------------------------------------
    def _load_arrow_from_s3(self, key: str) -> pa.Table:
        """Download a single S3 object and parse as Arrow or Parquet."""
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"].read()

        if key.endswith(".parquet"):
            return pq.read_table(pa.BufferReader(body))

        # Arrow file → try file format, then stream format
        buf = pa.BufferReader(body)
        try:
            reader = pa_ipc.open_file(buf)
            return reader.read_all()
        except pa.lib.ArrowInvalid:
            reader = pa_ipc.open_stream(buf)
            return reader.read_all()

    # ------------------------------------------------------------------
    def _table_to_df(self, table: pa.Table) -> pd.DataFrame:
        df = table.to_pandas()
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})
        df = df.dropna(subset=["phrase"])
        return df

    # ------------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Yield DataFrame batches across all shards.
        """
        logger.info(f"📥 Streaming from {len(self.files)} S3 shards…")

        for key in self.files:
            logger.info(f"→ Loading {key}")

            table = self._load_arrow_from_s3(key)
            df = self._table_to_df(table)

            total = len(df)
            num_batches = int(np.ceil(total / self.batch_size))

            for i in range(num_batches):
                batch_df = df.iloc[i * self.batch_size : (i + 1) * self.batch_size]
                logger.info(
                    f"  • Yield batch {i+1}/{num_batches} ({len(batch_df)} rows)"
                )
                yield batch_df
