def get_srgan_model():
    from models.SRGAN.SRGAN import SRGAN_model
    model = SRGAN_model(config_file_path="models/SRGAN/config.yaml")
    model.load_generator(checkpoint_path="models/SRGAN/srgan.ckpt")
    model.eval()
    return model