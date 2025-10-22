# main training script for the lab

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



if __name__ == '__main__':
    # need to take in arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--swiglu', '-s', action='store_true', help='Enable SwiGLU activation in the model')
    parser.add_argument('--rope', '-r', action='store_true', help='Enable Rotary Positional Embeddings in the model')
    parser.add_argument('--lw_scheduler', '-l', action='store_true', help='Use linear warmup scheduler instead of cosine')
    parser.add_argument('--cos_scheduler', '-c', action='store_true', help='Use cosine annealing scheduler')
    parser.add_argument('--rms_norm', '-r', action='store_true', help='Use RMSNorm instead of LayerNorm in the model')
    parser.add_argument('--max_iters', '-m', type=int, default=1000, help='Maximum number of training iterations')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training')
    parser.add_argument('--subset', action='store_true', help='Use a small subset of data to use for training')
    parser.add_argument('--model', '-M', type=str, default='gpt2', help='Model architecture to use (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--all_features', '-a', action='store_true', help='Enable all features (SwiGLU, RoPE, linear scheduler, RMSNorm)')
    args = parser.parse_args()

    # set args in a config dict
    config = {
        'swiglu': args.swiglu,
        'rope': args.rope,
        'linear_scheduler': args.linear_scheduler,
        'cosine_scheduler': args.cosine_scheduler,
        'rms_norm': args.rms_norm,
        'max_iters': args.max_iters,
        'batch_size': args.batch_size,
        'subset': args.subset,
        'model': args.model,
        'all_features': args.all_features
    }
    main(config)