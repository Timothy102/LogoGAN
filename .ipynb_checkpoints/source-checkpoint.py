from dataset import data
from model import *
from configuration import DATASET_PATH, LAST_EPOCH, EPOCHS


def main():
    # Create models
    gen_model = model.generator()
    gen_optimizer = model.generator_optimizer()
    dis_model = model.discriminator()
    dis_optimizer = model.discriminator_optimizer()

    # Define model checkpoint
    checkpoint, checkpoint_prefix = model.restore_checkpoint(gen_model, gen_optimizer, dis_model, dis_optimizer)

    # Initialize training pipeline.
    model.train(
        real_image_dataset=data,
        last_epoch=LAST_EPOCH,
        epochs=EPOCHS,
        gen_model=gen_model,
        gen_optimizer=gen_optimizer,
        dis_model=dis_model,
        dis_optimizer=dis_optimizer,
        checkpoint=checkpoint,
        checkpoint_prefix=checkpoint_prefix,
    )

    return 0


if __name__ == '__main__':
    main()