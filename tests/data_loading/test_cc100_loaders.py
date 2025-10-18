import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phrasely.data_loading.cc100_loader import CC100Loader
from phrasely.data_loading.cc100_offline_loader import CC100OfflineLoader


@pytest.fixture
def tmp_cc100_dir(tmp_path):
    """Create a small fake CC100 parquet dataset for offline testing."""
    df = pd.DataFrame({"text": [f"example phrase {i}" for i in range(200)]})
    path = tmp_path / "cc100_en_00.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    return tmp_path


def test_offline_loader_reads_and_samples(tmp_cc100_dir):
    loader = CC100OfflineLoader(
        arrow_dir=tmp_cc100_dir,
        language="en",
        max_phrases=50,
        seed=123,
    )
    df = loader.load()
    assert isinstance(df, pd.DataFrame)
    assert "phrase" in df.columns
    assert len(df) == 50  # sampled
    assert df["phrase"].notna().all()
    # deterministic sampling check
    df2 = loader.load()
    assert df.equals(df2)


def test_online_loader_has_expected_columns(monkeypatch):
    """Mock load_dataset to avoid huge downloads."""
    import phrasely.data_loading.cc100_loader as ccmod

    # fake dataset returned by Hugging Face
    fake_dataset = pd.DataFrame({"text": [f"sample {i}" for i in range(10)]})

    def fake_load_dataset(*args, **kwargs):
        class Dummy:
            def __getitem__(self, key):
                return fake_dataset[key]

            def to_pandas(self):
                return fake_dataset

            def __iter__(self):
                return iter(fake_dataset.to_dict(orient="records"))

            def __len__(self):
                return len(fake_dataset)

            def __array__(self):
                return fake_dataset

        return fake_dataset

    monkeypatch.setattr(ccmod, "load_dataset", fake_load_dataset)

    loader = CC100Loader(language="en", max_phrases=5)
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert "phrase" in df.columns
    assert len(df) == 5
    assert df["phrase"].iloc[0].startswith("sample")


def test_loaders_have_consistent_interface():
    """Ensure both loaders accept the same argument names."""
    import inspect

    online_args = set(inspect.signature(CC100Loader.__init__).parameters)
    offline_args = set(inspect.signature(CC100OfflineLoader.__init__).parameters)
    # ignore 'self'
    online_args.discard("self")
    offline_args.discard("self")

    # offline has arrow_dir extra, but all other args should overlap
    common = {"language", "max_phrases", "seed"}
    assert common.issubset(online_args)
    assert common.issubset(offline_args)
