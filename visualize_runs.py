OUTPUT_DIR="output/experiments/test_base"

def main():
    import os
    import torch
    from mingpt.trainer import Trainer

    # Load training losses
    losses_path = os.path.join(OUTPUT_DIR, 'train_losses.pt')
    if os.path.exists(losses_path):
        train_losses = torch.load(losses_path)
        print(f"Loaded training losses from {losses_path}")
    else:
        print(f"No training losses found at {losses_path}")
        return

    # Simple visualization using matplotlib
    import matplotlib.pyplot as plt

    # save to file
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, 'training_loss.png')
    plt.savefig(plot_path)
    print(f"Saved training loss plot to {plot_path}")

if __name__ == "__main__":
    main()