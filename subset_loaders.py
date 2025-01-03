import mnist_dataset
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import random

# Function to binarize images
def binarize_images(images, threshold=0.5):
    return (images > threshold).float()

# Function to create class-specific data loaders
def create_class_dataloaders(dataset, batch_size, num_classes=10, binarize=False, threshold=0.5):
    class_loaders = {}
    class_indices = defaultdict(list)

    # Step 1: Precompute indices for each class
    for i, (_, label) in enumerate(dataset):
        class_indices[label].append(i)
    
    # Step 2: Create DataLoaders for each class
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        
        # Create a Subset of the dataset with only the current class
        class_subset = Subset(dataset, indices)
        
        # Create a custom dataset that applies binarization if required
        class_dataset = BinarizedDataset(class_subset, binarize, threshold) if binarize else class_subset
        
        # Create a DataLoader for this class
        class_loader = DataLoader(class_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        class_loaders[class_idx] = class_loader
    
    return class_loaders

# Custom Dataset class that can apply binarization
class BinarizedDataset(Dataset):
    def __init__(self, dataset, binarize=False, threshold=0.5):
        self.dataset = dataset
        self.binarize = binarize
        self.threshold = threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.binarize:
            image = binarize_images(image, self.threshold)
        return image, label

def predict_and_categorize(model, data_loader, binarize=False, threshold=0.5):
    all_images = []
    all_labels = []
    all_predictions = []

    # Ensure model is in evaluation mode
    model.diff_logic_model.eval()

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_inputs, batch_outputs in tqdm(data_loader, desc="Predicting"):
            batch_inputs, batch_outputs = batch_inputs.to('cuda'), batch_outputs.to('cuda')

            if binarize:
                batch_inputs = (batch_inputs > threshold).float()

            # Forward pass to get predictions
            outputs = model.diff_logic_model(batch_inputs)

            # Get the predicted class (index of the maximum logit)
            _, predicted = torch.max(outputs.data, 1)

            all_images.append(batch_inputs.cpu())
            all_labels.append(batch_outputs.cpu())
            all_predictions.append(predicted.cpu())

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    return all_images, all_labels, all_predictions


def create_tn_tp_fn_fp_datasets(images, labels, predictions):
    t_mask = (predictions == labels)
    f_mask = (predictions != labels)
    
    t_dataset = TensorDataset(images[t_mask], labels[t_mask])
    f_dataset = TensorDataset(images[f_mask], labels[f_mask])
    
    return {
        'T': t_dataset,
        'F': f_dataset,
    }

def create_class_specific_datasets(images, labels, predictions, num_classes=10):
    datasets = {
        'TP_class': {},
        'FP_class': {},
        'TN_class': {},
        'FN_class': {}
    }
    
    for class_idx in range(num_classes):
        # TP_class
        tp_mask = (predictions == class_idx) & (labels == class_idx)
        datasets['TP_class'][class_idx] = TensorDataset(images[tp_mask], labels[tp_mask])

        # TN_class
        tn_mask = (predictions != class_idx) & (labels != class_idx)
        datasets['TN_class'][class_idx] = TensorDataset(images[tn_mask], labels[tn_mask])
        
        # FP_class
        fp_mask = (predictions == class_idx) & (labels != class_idx)
        datasets['FP_class'][class_idx] = TensorDataset(images[fp_mask], labels[fp_mask])
        
        # FN_class
        fn_mask = (predictions != class_idx) & (labels == class_idx)
        datasets['FN_class'][class_idx] = TensorDataset(images[fn_mask], labels[fn_mask])
    
    return datasets


'''

# Create the datasets
general_datasets = create_tn_tp_fn_fp_datasets(all_images, all_labels, all_predictions)
class_specific_datasets = create_class_specific_datasets(all_images, all_labels, all_predictions)

# Fix random seeds for reproducibility
torch.manual_seed(42)            
torch.cuda.manual_seed(42)        
np.random.seed(42)                
random.seed(42)                   
# If using CUDA:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 256 # this can be tuned as well
binarize = True  # Set this to True if you want to binarize the images
threshold = 0.5  # Set the threshold for binarization

# Load the full datasets
train_dataset = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=True)
test_dataset  = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=True)

# Create class-specific data loaders for training and testing sets
train_class_loaders = create_class_dataloaders(train_dataset, batch_size, binarize=binarize, threshold=threshold)
test_class_loaders = create_class_dataloaders(test_dataset, batch_size, binarize=binarize, threshold=threshold)

# Create full dataset loaders
train_dataset = BinarizedDataset(train_dataset, binarize, threshold)
test_dataset = BinarizedDataset(test_dataset, binarize, threshold)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

# Predict the TN, TP, FN, FP 
all_images, all_labels, all_predictions = predict_and_categorize(model, test_loader, binarize=binarize, threshold=threshold)

# Example of how to use these loaders:
#print("Full training dataset:")
#for batch in tqdm(train_loader, desc="Processing full training set"):
#    # Process full dataset
#    pass

#print("\nClass-specific datasets:")
#for class_idx, loader in train_class_loaders.items():
#    print(f"Processing class {class_idx}")
#    for batch in tqdm(loader, desc=f"Processing class {class_idx}"):
#        # Process class-specific data
#        pass

'''
