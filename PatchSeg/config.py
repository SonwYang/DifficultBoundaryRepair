class Config(object):
    ### pixel_values = [104.00699, 116.66877, 122.67892, 137.86]
    img_width = 384
    img_height = 384
    train_root = './data/patches/train_images'
    valid_output_dir = 'valid_temp'
    resume = 'model.pth'

    ### hyper parameters
    batch_size = 32
    num_workers = 0
    num_epochs = 1000
    model_output = 'ckpts_unetpp'
