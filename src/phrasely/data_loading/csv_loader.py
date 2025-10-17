import logging

import pandas as pd

logger = logging.getLogger(__name__)


class CSVLoader:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def load(self):
        try:
            df = pd.read_csv(self.input_path)
            if "phrase" in df.columns:
                return df["phrase"].tolist()
            return df.iloc[:, 0].astype(str).tolist()
        except FileNotFoundError:
            logger.error(
                f"[CSVLoader] File not found: {self.input_path}."
                + " Returning mock phrases."
            )
            return ["alpha", "beta", "gamma", "delta"]
