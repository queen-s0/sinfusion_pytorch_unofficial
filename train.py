import argparse
from sinfusion import Unet, GaussianDiffusion, Trainer, CropImageDataset


def main(args):

    model = Unet(dim=64)

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size, #128
        timesteps=args.timesteps,   # number of steps 50
        sampling_timesteps=args.sampling_timesteps, # = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=args.loss_type # = 'l2'            # L1 or L2
    )

    dataset = CropImageDataset(args.image_path, args.image_size)

    trainer = Trainer(diffusion, dataset)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('-T', '--timesteps', default=50)
    parser.add_argument('--sampling_timesteps', default=50)
    parser.add_argument('-l', '--loss_type', default='l2')
    parser.add_argument('--image_size', default=128)
    args = parser.parse_args()
    
    main(args)