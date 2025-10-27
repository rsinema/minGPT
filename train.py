# main training script for the lab

import os
from dotenv import load_dotenv

def build_configs(config: dict):
    # build model config
    from mingpt.model import GPT
    model_config = GPT.get_default_config()
    model_config.merge_from_dict(config)

    # build trainer config
    from mingpt.trainer import Trainer
    trainer_config = Trainer.get_default_config()
    trainer_config.merge_from_dict(config)
    return model_config, trainer_config


def main(config: dict):
    import torch

    from mingpt.trainer import Trainer
    from mingpt.model import GPT
    from mingpt.dataset import JSONLDataset

    model_config, trainer_config = build_configs(config)

    file_path = os.getenv('DATASET_PATH', '/nobackup/autodelete/usr/rsinema/pile_data_10_min.jsonl')
    output_dir = os.getenv('OUTPUT_DIR', 'output').join(config.get('exp_name', 'default_exp'))
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = JSONLDataset(file_path, split='train', test_size=10)
    val_dataset = JSONLDataset(file_path, split='val', test_size=10)

    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()

    model = GPT(model_config)

    trainer = Trainer(trainer_config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            # save losses to a file using torch.save
            torch.save(trainer.losses, os.path.join(output_dir, 'train_losses.pt'))

            if trainer.iter_num % 1000 == 0:
                # save the model checkpoint
                ckpt_path = os.path.join(output_dir, f'model_iter_{trainer.iter_num}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved model checkpoint to {ckpt_path}")

    trainer.run()

if __name__ == '__main__':
    # need to take in arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--swiglu', '-s', action='store_true', help='Enable SwiGLU activation in the model')
    parser.add_argument('--rope', '-r', action='store_true', help='Enable Rotary Positional Embeddings in the model')
    parser.add_argument('--lw_scheduler', '-l', action='store_true', help='Use linear warmup scheduler instead of cosine')
    parser.add_argument('--cos_scheduler', '-c', action='store_true', help='Use cosine annealing scheduler')
    parser.add_argument('--rms_norm', '-R', action='store_true', help='Use RMSNorm instead of LayerNorm in the model')
    parser.add_argument('--max_iters', '-m', type=int, default=1000, help='Maximum number of training iterations')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training')
    parser.add_argument('--subset', action='store_true', help='Use a small subset of data to use for training')
    parser.add_argument('--model', '-M', type=str, default='gpt2', help='Model architecture to use (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--all_features', '-a', action='store_true', help='Enable all features (SwiGLU, RoPE, linear scheduler, RMSNorm)')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Name of the experiment for logging purposes')
    args = parser.parse_args()

    # set args in a config dict
    config = {
        'swiglu': args.swiglu,
        'rope': args.rope,
        'lw_scheduler': args.lw_scheduler,
        'cos_scheduler': args.cos_scheduler,
        'rms_norm': args.rms_norm,
        'max_iters': args.max_iters,
        'batch_size': args.batch_size,
        'subset': args.subset,
        'model': args.model,
        'all_features': args.all_features,
        'exp_name': args.exp_name
    }
    main(config)