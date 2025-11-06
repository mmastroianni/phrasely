import logging
import io
from typing import Generator, Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

logger = logging.getLogger(__name__)


class CC100S3Loader:
    """
    Streams CC100 Arrow/Parquet shards directly from S3 *without* saving to disk.

    Matches the interface & behavior of `CC100OfflineLoader`, but loads files via boto3.

    Parameters
    ----------
    bucket : str
        S3 bucket name (e.g. "phrasely-data-mastroianni")
    prefix : str
        Prefix containing Arrow/Parquet files (e.g. "cc100/")
    language : str, default="en"
        Optional language filter in filename.
    max_files : int, optional
        Cap number of shards (debug/testing).
    batch_size : int, default=20_000
        Number of rows yielded per mini-batch.
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

        # discover files
        resp = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/"
        )
        all_files = [obj["Key"] for obj in resp.get("Contents", [])
                     if obj["Key"].endswith((".arrow", ".parquet"))]

        # optional language filter
        if language:
            all_files = [k for k in all_files if language.lower() in k.lower()]

        if max_files is not None and len(all_files) > max_files:
            logger.warning(f"Limiting to first {max_files} of {len(all_files)} files.")
            all_files = all_files[:max_files]

        if not all_files:
            raise FileNotFoundError(
                f"No matching Arrow/Parquet files under s3://{bucket}/{prefix}"
            )

        self.files = sorted(all_files)
        logger.info(f"Found {len(self.files)} S3 shards.")

    # -------------------------------------------------------------
    def _load_arrow_table(self, key: str) -> pa.Table:
        """Download Arrow file into memory and return a PyArrow Table."""
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"].read()

        buf = pa.BufferReader(body)
        try:
            reader = pa_ipc.open_file(buf)
        except pa.lib.ArrowInvalid:
            reader = pa_ipc.open_stream(buf)

        return reader.read_all()

    # -------------------------------------------------------------
    def _table_to_df(self, table: pa.Table) -> pd.DataFrame:
        df = table.to_pandas()
        if "text" in df.columns:
            df = df.rename(columns={"text": "phrase"})
        return df.dropna(subset=["phrase"])

    # -------------------------------------------------------------
    def stream_load(self) -> Generator[pd.DataFrame, None, None]:
        """
        Yield DataFrame mini-batches sequentially across all S3 shards.
        """
        logger.info(f"Streaming from {len(self.files)} shards in s3://{self.bucket}/{self.prefix}")

        for key in self.files:
            logger.info(f"Downloading {key} ...")
            table = self._load_arrow_table(key)
            df = self._table_to_df(table)

            n = len(df)
            num_batches = int(np.ceil(n / self.batch_size))

            for i in range(num_batches):
                batch_df = df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                logger.info(f"Yielding batch {i+1}/{num_batches} of {key} ({len(batch_df)} rows)")
                yield batch_df
