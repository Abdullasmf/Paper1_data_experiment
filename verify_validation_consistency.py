"""
Verification script to ensure validation data consistency across different training data percentages.
This script validates that:
1. The same validation indices are used regardless of training data percentage
2. Training data subsampling is reproducible
3. Validation set remains unchanged
"""

import random

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    random.seed(seed)

def train_test_split_simple(indices, test_size=0.2, random_state=42):
    """Simple implementation of train_test_split for verification"""
    random.seed(random_state)
    n = len(indices)
    indices_copy = indices.copy()
    random.shuffle(indices_copy)
    split_idx = int(n * (1 - test_size))
    train_idx = indices_copy[:split_idx]
    val_idx = indices_copy[split_idx:]
    return train_idx, val_idx

def simulate_data_split(n_geoms: int, data_perc: int):
    """Simulate the data splitting logic from Training_script.py"""
    set_seed(42)
    
    # Step 1: Create indices
    idxs = list(range(n_geoms))
    
    # Step 2: Split into train/val
    train_idx, val_idx = train_test_split_simple(idxs, test_size=0.2, random_state=42)
    
    print(f"\nOriginal split (data_perc={data_perc}%):")
    print(f"  Total geometries: {n_geoms}")
    print(f"  Training geometries: {len(train_idx)}")
    print(f"  Validation geometries: {len(val_idx)}")
    print(f"  Validation indices (first 10): {sorted(val_idx)[:10]}")
    
    # Step 3: Subsample training data if needed
    if data_perc < 100:
        set_seed(42)
        n_train = len(train_idx)
        n_subsample = max(1, int(n_train * data_perc / 100))
        train_idx_subsampled = random.sample(train_idx, n_subsample)
        print(f"\nAfter subsampling to {data_perc}%:")
        print(f"  Training geometries: {len(train_idx_subsampled)} (from {n_train})")
        print(f"  Training indices (first 10): {sorted(train_idx_subsampled)[:10]}")
        train_idx = train_idx_subsampled
    
    return train_idx, val_idx

def main():
    """Run validation consistency checks"""
    print("="*80)
    print("Validation Data Consistency Verification")
    print("="*80)
    
    # Assume we have 100 geometries (similar to actual dataset size)
    n_geoms = 100
    
    # Test with different data percentages
    test_percentages = [40, 60, 100]
    
    val_sets = {}
    train_sets = {}
    
    for data_perc in test_percentages:
        train_idx, val_idx = simulate_data_split(n_geoms, data_perc)
        val_sets[data_perc] = set(val_idx)
        train_sets[data_perc] = set(train_idx)
    
    # Verification 1: Check that validation sets are identical
    print("\n" + "="*80)
    print("VERIFICATION 1: Validation Set Consistency")
    print("="*80)
    
    val_set_40 = val_sets[40]
    all_same = True
    for perc in test_percentages:
        if val_sets[perc] == val_set_40:
            print(f"✓ Validation set at {perc}% matches baseline (40%)")
        else:
            print(f"✗ Validation set at {perc}% differs from baseline (40%)")
            all_same = False
    
    if all_same:
        print("\n✓✓✓ SUCCESS: All validation sets are identical!")
    else:
        print("\n✗✗✗ FAILURE: Validation sets differ across experiments!")
    
    # Verification 2: Check that training sets are different and subset relationships are correct
    print("\n" + "="*80)
    print("VERIFICATION 2: Training Set Subsampling")
    print("="*80)
    
    print(f"Training set sizes:")
    for perc in test_percentages:
        print(f"  {perc}%: {len(train_sets[perc])} geometries")
    
    # Verify that subsampled training sets are different from full set
    if train_sets[40] != train_sets[100]:
        print("✓ 40% training set is different from 100% training set (expected)")
    else:
        print("✗ 40% training set is same as 100% training set (unexpected)")
    
    # Verify that subsampled training set is a subset of full training set
    original_train_80 = set(range(n_geoms)) - val_set_40  # Original 80% training set
    if train_sets[40].issubset(original_train_80):
        print("✓ 40% training set is a subset of the original training split")
    else:
        print("✗ 40% training set is NOT a subset of the original training split")
    
    # Verification 3: Check reproducibility
    print("\n" + "="*80)
    print("VERIFICATION 3: Reproducibility Check")
    print("="*80)
    
    # Run the same split twice
    train_idx_1, val_idx_1 = simulate_data_split(n_geoms, 40)
    train_idx_2, val_idx_2 = simulate_data_split(n_geoms, 40)
    
    if set(val_idx_1) == set(val_idx_2):
        print("✓ Validation indices are reproducible")
    else:
        print("✗ Validation indices are NOT reproducible")
    
    if set(train_idx_1) == set(train_idx_2):
        print("✓ Training indices are reproducible")
    else:
        print("✗ Training indices are NOT reproducible")
    
    # Verification 4: Ensure no overlap between train and validation
    print("\n" + "="*80)
    print("VERIFICATION 4: Train/Validation Separation")
    print("="*80)
    
    for perc in test_percentages:
        overlap = train_sets[perc].intersection(val_sets[perc])
        if len(overlap) == 0:
            print(f"✓ No overlap between train and validation at {perc}%")
        else:
            print(f"✗ Found {len(overlap)} overlapping indices at {perc}%")
    
    print("\n" + "="*80)
    print("Verification Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
