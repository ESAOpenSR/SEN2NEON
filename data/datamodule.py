from torch.utils.data import DataLoader, Subset
import pandas as pd
from data.dataset import SEN2NEON
import torch

class SEN2NEONDataModule:
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        *,
        split: str | None = "val",     # set to None to use all rows
        subset: int | None = None,     # take first N after filtering
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        # CSVPairedTiffDataset kwargs:
        crop_size_lr: int | None = None,
        dtype: torch.dtype = torch.float32,
        allow_nan: bool = False,
        pattern_check: bool = True,
    ):
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.split = split
        self.subset = subset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.ds_kwargs = dict(
            crop_size_lr=crop_size_lr,
            dtype=dtype,
            allow_nan=allow_nan,
            pattern_check=pattern_check,
        )

        self.predict_dataset = None

    def setup(self, stage: str | None = None):
        ds = SEN2NEON(
            csv_path=self.csv_path,
            root_dir=self.root_dir,
            **self.ds_kwargs,
        )

        # optional split filtering (only if CSV has a 'split' column)
        indices = list(range(len(ds)))
        if self.split is not None and "split" in ds.df.columns:
            indices = ds.df.index[ds.df["split"].astype(str) == str(self.split)].tolist()

        # optional subsetting
        if self.subset is not None and self.subset > 0:
            indices = indices[: self.subset]

        self.predict_dataset = Subset(ds, indices)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        
if __name__ == "__main__":
    dm = SEN2NEONDataModule(
    csv_path="/data3/SEN2NEON/sen2neon_metadata.csv",
    root_dir="/data3/SEN2NEON",
    split="val",            # or None to ignore CSV split
    batch_size=4,
    crop_size_lr=64,
    )
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))

