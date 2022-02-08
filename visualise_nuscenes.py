from fiery.trainer import TrainingModule
from fiery.data import prepare_dataloaders
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from visualise import plot_prediction

import torch
from argparse import ArgumentParser
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

def vis_nusc(checkpoint_path, dataroot, version, output_file):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)
    
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    
    out_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 20, (861, 200))


    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output = model(image, intrinsics, extrinsics, future_egomotion,
                           future_distribution_inputs, noise=noise)
            figure_numpy = plot_prediction(image, output, trainer.cfg)
            plt.imshow(figure_numpy)
            frame = cv2.cvtColor(figure_numpy, cv2.COLOR_RGB2BGR)
            print(frame.shape)
            out_video.write(frame)
            
    out_video.release()
    print("Done")

if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery visualisation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='data/nuscenes', type=str, help='path to root directory of nuScenes dataset')
    parser.add_argument('--version', default='mini', type=str, help='version of nuScenes dataset')
    parser.add_argument('--output-file', default='output_fiery.mp4', type=str, help='output video file')

    args = parser.parse_args()

    vis_nusc(args.checkpoint, args.dataroot, args.version, args.output_file)