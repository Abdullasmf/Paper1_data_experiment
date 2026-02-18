"""
Example script to load and plot loss history from trained models.
This demonstrates how to access the training and validation loss history
that is saved in the model checkpoint files.
"""

def plot_loss_history_example():
    """
    Example code showing how to load and plot loss history from a trained model.
    
    Note: This is demonstration code. To actually run it, you need:
    1. A trained model checkpoint (.pt file)
    2. matplotlib installed
    3. torch installed
    """
    
    # Example code (commented out since we don't have actual trained models yet)
    example_code = '''
import torch
import matplotlib.pyplot as plt

# Load the trained model checkpoint
model_path = 'PointNetMLPJoint/Trained_models/L-_40p_pnmlp_12345678.pt'
checkpoint = torch.load(model_path, map_location='cpu')

# Extract loss histories
train_losses = checkpoint['train_loss_history']
val_losses = checkpoint['val_loss_history']

print(f"Model trained for {len(train_losses)} epochs")
print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
print(f"Final training loss: {train_losses[-1]:.6f}")
print(f"Final validation loss: {val_losses[-1]:.6f}")

# Plot loss history
fig, ax = plt.subplots(figsize=(12, 6))

epochs = range(1, len(train_losses) + 1)
ax.plot(epochs, train_losses, label='Training Loss', linewidth=2, alpha=0.8)
ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Training and Validation Loss History', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Optionally use log scale if losses vary greatly
# ax.set_yscale('log')

plt.tight_layout()
plt.savefig('loss_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare multiple models
model_paths = [
    'PointNetMLPJoint/Trained_models/L-_40p_pnmlp_12345678.pt',
    'PointNetMLPJoint/Trained_models/L-_100p_pnmlp_12345678.pt',
    'VanillaDeepONet/Trained_models/L-_40p_pnmlp_87654321.pt',  # Note: actual filenames will vary
    'VanillaDeepONet/Trained_models/L-_100p_pnmlp_87654321.pt',
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for model_path in model_paths:
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        val_losses = ckpt['val_loss_history']
        
        # Extract data percentage and model type from path
        parts = model_path.split('/')
        filename = parts[-1]
        label = filename.replace('.pt', '')
        
        epochs = range(1, len(val_losses) + 1)
        ax1.plot(epochs, val_losses, label=label, linewidth=2, alpha=0.7)
    except FileNotFoundError:
        print(f"Model not found: {model_path}")

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation MSE Loss', fontsize=12)
ax1.set_title('Validation Loss Comparison', fontsize=14)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Bar chart of final validation losses
final_losses = []
labels = []
for model_path in model_paths:
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        final_losses.append(ckpt['best_val_loss'])
        
        parts = model_path.split('/')
        filename = parts[-1]
        labels.append(filename.replace('.pt', ''))
    except FileNotFoundError:
        pass

ax2.bar(range(len(final_losses)), final_losses)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Best Validation MSE', fontsize=12)
ax2.set_title('Best Validation Loss Comparison', fontsize=14)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
'''
    
    print("Example Code for Loading and Plotting Loss History")
    print("=" * 80)
    print(example_code)
    print("=" * 80)
    
    print("\nThe loss history is automatically saved in each model checkpoint with keys:")
    print("  - 'train_loss_history': List of training MSE per epoch")
    print("  - 'val_loss_history': List of validation MSE per epoch")
    print("\nThese can be used to:")
    print("  1. Plot training curves")
    print("  2. Compare convergence across different data percentages")
    print("  3. Analyze overfitting (train vs validation gap)")
    print("  4. Compare different model architectures")

if __name__ == "__main__":
    plot_loss_history_example()
