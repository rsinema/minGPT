OUTPUT_DIR="output/experiments/"

def main():
    import os
    import torch

    # for each directory in OUTPUT_DIR, load the training losses and visualize them
    for dir in os.listdir(OUTPUT_DIR):
        dir_path = os.path.join(OUTPUT_DIR, dir)
        print(f"Processing directory: {dir_path}")
        if os.path.isdir(dir_path):
            losses_path = os.path.join(dir_path, 'train_losses.pt')
            if os.path.exists(losses_path):
                train_losses = torch.load(losses_path)
            else:
                print(f"No training losses found at {losses_path}")
                continue

            # Simple visualization using matplotlib
            import matplotlib.pyplot as plt

            # get the directory name for title
            dir_name = os.path.basename(dir_path)

            # save to file
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Training Loss for {dir_name}')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(dir_path, 'training_loss.png')
            plt.savefig(plot_path)
            print(f"Saved training loss plot to {plot_path}")

if __name__ == "__main__":
    main()