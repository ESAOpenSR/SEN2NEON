from data.sen2neon_ds import SEN2NEON, SEN2NEONDataModule

datamodule = SEN2NEONDataModule(
    lr_dir="/data3/SEN2NEON/processed/neon_10m_linearized",
    hr_dir="/data3/SEN2NEON/processed/neon_2.5m_linearized",
    pattern="*2019-06-15*T1*",
    batch_size=4,
    allow_nan=False,
    pin_memory=True,
)
datamodule.setup(stage="predict") # set up datamodule for prediction
loader = datamodule.predict_dataloader() # get the prediction dataloader
batch = next(iter(loader)) # get a batch from the dataloader
print(batch["lr"].shape, batch["hr"].shape, batch["name"]) # print shapes of lr, hr, and names in the batch
