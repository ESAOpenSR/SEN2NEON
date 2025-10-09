# Package Imports
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# local imports
from models.SRGAN.model_blocks import Generator
from models.SRGAN.utils import normalise_10k, histogram_match

#############################################################################################################
# Build PL MODEL


class SRGAN_model(pl.LightningModule):

    def __init__(self, config_file_path="config.yaml"):
        super(SRGAN_model, self).__init__()

        # get config file
        self.config = OmegaConf.load(config_file_path)

        # Generator
        self.generator = Generator(in_channels = self.config.Model.in_bands,
                        large_kernel_size=self.config.Generator.large_kernel_size,
                        small_kernel_size=self.config.Generator.small_kernel_size,
                        n_channels=self.config.Generator.n_channels,
                        n_blocks=self.config.Generator.n_blocks,
                        scaling_factor=self.config.Generator.scaling_factor)

    def forward(self,lr_imgs):
        # if MISR, perform Fusion first
        if self.config.SR_type=="MISR":
            lr_imgs = self.fusion(lr_imgs)
        # perform generative Step
        sr_imgs = self.generator(lr_imgs)
        return(sr_imgs)
    

    @torch.no_grad()
    def predict_step(self,lr_imgs):
        """
        This function is for the prediction in the Deployment stage, therefore
        the normalization and denormalization needs to happen here.
        Input:
            - unnormalized lLR imgs
        Output:
            - normalized SR images
        Info:
            - This function currently only performs SISR SR
        """

        need_to_normalize = lr_imgs.max()>2.

        # move to GPU if possible
        lr_imgs = lr_imgs.to(self.device)
        # normalize images
        #lr_imgs = normalise_10k(lr_imgs,stage="norm")
        # preform SR
        with torch.no_grad():
            sr_imgs = self.generator(lr_imgs)
        # denormalize images
        #sr_imgs = normalise_10k(sr_imgs,stage="denorm")
        # histogram match to also encoded LR images
        sr_imgs = histogram_match(lr_imgs,sr_imgs)
        # move to CPU
        sr_imgs = sr_imgs.cpu().detach()
        return sr_imgs

    def load_generator(self,checkpoint_path=None):
        """
        Load pretrained weights into the generator
        If no checkpoint path is provided, load the default weights
        """
        state_dict = torch.load(checkpoint_path)["state_dict"]
        self.load_state_dict(state_dict,strict=False)
        print(f"Loaded Generator Weights from {checkpoint_path}")
        del state_dict

if __name__=="__main__":       
    model = SRGAN_model(config_file_path="models/SRGAN/config.yaml")
    model.load_generator(checkpoint_path="models/SRGAN/srgan.ckpt")

    # test model
    lr = torch.randn(1,6,64,64)
    sr = model.predict_step(lr)

