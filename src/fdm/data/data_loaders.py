from torch.utils.data import random_split, DataLoader

def get_data_loaders(dataset, batch_size, collate_fn):
    train_size = int(0.7 * len(dataset))
    test_and_val_size = len(dataset) - train_size
    val_size = int(0.5 * test_and_val_size)
    test_size = test_and_val_size - val_size

    train_dataset, val_dataset, test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

