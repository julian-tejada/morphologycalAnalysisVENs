#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:09:17 2024

@author: lrmercadod
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_fscore_support
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim
import cv2
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the metadata file
metadata_path = r'/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/neuromorpho_data/MetaData_swc_files.csv'
metadata = pd.read_csv(metadata_path)

# Filter metadata for human species and relevant columns
filtered_metadata = metadata[metadata['species'] == 'human'].copy()

# Map cell types to numerical labels
cell_type_mapping = {'von Economo neuron': 1, 'pyramidal': 0}
filtered_metadata['label'] = filtered_metadata['cell_type1'].map(cell_type_mapping)

# Select relevant columns
relevant_columns = ['file_name', 'label', 'Soma_surface', 'Surface.Total_sum', 'Volume.Total_sum',
                    'N_stems.Total_sum', 'N_bifs.Total_sum', 'N_branch.Total_sum', 'Width.Average',
                    'Height.Maximum', 'Depth.Maximum', 'Diameter.Average', 'EucDistance.Maximum',
                    'EucDistance.Total_sum', 'PathDistance.Total_sum', 'Branch_Order.Maximum',
                    'Contraction.Total_sum', 'Contraction.Maximum', 'Fragmentation.Average', 
                    'Fragmentation.Maximum', 'Partition_asymmetry.Total_sum', 'Pk_classic.Average',
                    'Pk_classic.Maximum', 'Bif_ampl_local.Maximum', 'Bif_ampl_local.Total_sum',
                    'Fractal_Dim.Total_sum', 'Fractal_Dim.Average', 'Bif_ampl_remote.Average',
                    'Bif_ampl_remote.Maximum', 'Length.Total_sum', 'Length.Average']
filtered_metadata = filtered_metadata[relevant_columns]

# Separate features and labels
X = filtered_metadata.drop(columns=['file_name', 'label'])
y = filtered_metadata['label']

# Remove constant features
X = X.loc[:, (X != X.iloc[0]).any()]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using ANOVA F-test
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X_scaled, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

# Update filtered_metadata with selected features
filtered_metadata = filtered_metadata[['file_name', 'label'] + selected_features.tolist()]

# Step 2: Graph Construction with Enhanced Features
def read_swc(file_path):
    data = np.loadtxt(file_path)
    node_features = data[:, 2:5]  # X, Y, Z coordinates
    node_regions = data[:, 1]  # Node type or region information
    edge_index = []

    index_mapping = {int(row[0]): idx for idx, row in enumerate(data)}
    for row in data:
        parent_index = int(row[6])
        if parent_index != -1:  # No edge for the root node
            edge_index.append([index_mapping[parent_index], index_mapping[int(row[0])]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index, node_regions=torch.tensor(node_regions, dtype=torch.long))

def load_swc_files_from_directories(directories):
    graphs = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.swc'):
                file_path = os.path.join(directory, filename)
                graph_data = read_swc(file_path)
                graphs.append((filename, graph_data))
    return graphs

# List of directories containing SWC files
directories = [
    r"/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/neuromorpho_data/SWC_files/jacobs/CNG version",
    r"/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/neuromorpho_data/SWC_files/vdheuvel/CNG version",
    r"/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/neuromorpho_data/SWC_files/zagreb/CNG version",
    r"/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/neuromorpho_data/SWC_files/allman/CNG version"
]

# Load the SWC files and convert them to graphs
swc_graphs = load_swc_files_from_directories(directories)

# Combine graph data with metadata
combined_data = []
for filename, graph_data in swc_graphs:
    metadata_row = filtered_metadata[filtered_metadata['file_name'] == filename]
    if not metadata_row.empty:
        metadata_values = metadata_row.iloc[0, 1:].values.astype(np.float32)  # Get all metadata values except filename
        graph_data.metadata = torch.tensor(metadata_values[1:], dtype=torch.float)  # Exclude the label
        graph_data.label = torch.tensor(int(metadata_values[0]), dtype=torch.long)  # The label
        graph_data.file_name = filename
        combined_data.append(graph_data)

# Check combined data
len(combined_data)

# Define the visualize_swc function with color coding
def visualize_swc(graph_data, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract node features (X, Y, Z coordinates)
    node_positions = graph_data.x.numpy()
    node_regions = graph_data.node_regions.numpy()
    
    # Plot the nodes with color coding
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], c=node_regions, cmap='viridis', s=10)
    
    # Extract edge indices
    edge_index = graph_data.edge_index.numpy()
    
    # Plot the edges
    for edge in edge_index.T:
        start, end = edge
        ax.plot(
            [node_positions[start, 0], node_positions[end, 0]],
            [node_positions[start, 1], node_positions[end, 1]],
            [node_positions[start, 2], node_positions[end, 2]],
            color='black'
        )

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Region')

    # Set labels and save the figure
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)

# Visualize and save images for each SWC graph using Plotly
image_save_dir = '/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/ImagesNeuronas/'
os.makedirs(image_save_dir, exist_ok=True)

# Mapping between image names and original neuron file names
image_name_to_file_name = {}

for i, graph_data in enumerate(combined_data):
    original_name = graph_data.file_name.split('.')[0]  # Use the original neuron name
    image_save_path = os.path.join(image_save_dir, f"{original_name}.png")
    if not os.path.exists(image_save_path):
        visualize_swc(graph_data, image_save_path)
    image_name_to_file_name[f"{original_name}.png"] = graph_data.file_name

class NeuronImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Convert image to RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define constants
num_epochs = 10
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Create lists of image paths and labels
image_paths = [os.path.join(image_save_dir, f"{graph_data.file_name.split('.')[0]}.png") for graph_data in combined_data]
labels = [graph_data.label.item() for graph_data in combined_data]
file_names = [graph_data.file_name for graph_data in combined_data]

# Ensure that the lengths of image_paths and labels match
assert len(image_paths) == len(labels), "Mismatch between image paths and labels length"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cross-validation setup
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds)

# Initialize lists to store metrics across folds
all_fold_metrics = {
    'train_losses': [],
    'train_accuracies': [],
    'val_losses': [],
    'val_accuracies': [],
    'precisions': [],
    'recalls': [],
    'fscores': [],
    'roc_aucs': [],
    'fprs': [],  # Collect FPRs for ROC curve
    'tprs': [],  # Collect TPRs for ROC curve
    'conf_matrices': [],
}

# Function to apply masking to an image
def apply_mask(image, mask_type='soma'):
    image = np.array(image)
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if mask_type == 'soma':
        cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)
    elif mask_type == 'dendrites':
        mask = cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)
        mask = cv2.bitwise_not(mask)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return Image.fromarray(masked_image)

# Implementing Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()

        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[:, target_class]
        class_score.backward(retain_graph=True)

        # Generate CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
    
        # Fix normalization issue (avoid division by zero)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_min != cam_max:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam

# Function to calculate and store fprs and tprs for each fold
def compute_roc_curve(all_labels, all_preds):
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    return fpr, tpr

def apply_gradcam(image_path, model, gradcam, transform, original_name):
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate_cam(input_image)
    cam = cv2.resize(cam, (224, 224))

    # Normalize and create the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize
    heatmap = np.float32(heatmap) / 255

    # Load the original image and convert to float
    original_image = np.array(image.resize((224, 224))).astype(np.float32) / 255

    # Overlay the heatmap on the original image
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    return overlay, heatmap, original_name

def save_comparison_image(original_image_path, overlay, heatmap, original_name, save_dir):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (224, 224))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(overlay)
    axs[1].set_title('Grad-CAM Overlay')
    axs[1].axis('off')

    im = axs[2].imshow(heatmap, cmap='jet')
    axs[2].set_title('Heatmap')
    axs[2].axis('off')

    cbar = plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
    cbar.set_label('Importance')

    plt.suptitle(f'Comparison for {original_name}')
    comparison_save_path = os.path.join(save_dir, f"comparison_{original_name}.png")
    plt.savefig(comparison_save_path, dpi=300)
    plt.close()

# Cross-validation loop
metrics_per_fold = []  # To store metrics for each fold
avg_metrics = {  # To store the average metrics for all folds
    'train_losses': [],
    'train_accuracies': [],
    'val_losses': [],
    'val_accuracies': [],
    'precisions': [],
    'recalls': [],
    'fscores': [],
    'roc_aucs': [],
    'conf_matrices': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(np.array(image_paths), np.array(labels))):
    print(f"Starting fold {fold + 1}/{num_folds}")
    
    # Prepare data loaders for the current fold
    train_image_paths = [image_paths[i] for i in train_idx]
    val_image_paths = [image_paths[i] for i in val_idx]
    train_labels_fold = [labels[i] for i in train_idx]
    val_labels_fold = [labels[i] for i in val_idx]

    # Save the validation data
    validation_data = pd.DataFrame({
        'image_paths': val_image_paths,
        'labels': val_labels_fold,
        'file_names': [file_names[i] for i in val_idx]
    })
    validation_data.to_csv(f'validation_neurons_fold_{fold + 1}.csv', index=False)

    # Create datasets and data loaders
    train_dataset = NeuronImageDataset(train_image_paths, train_labels_fold, transform=transform)
    val_dataset = NeuronImageDataset(val_image_paths, val_labels_fold, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load and modify VGG16 model
    model = models.vgg16_bn(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize lists for storing misclassified and correctly classified neurons
    misclassified_neurons = []
    correct_classified_neurons = []

    # Training and validation loop for each fold
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, train_labels_batch in tqdm(train_loader):
            images, train_labels_batch = images.to(device), train_labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, train_labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += train_labels_batch.size(0)
            correct += (predicted == train_labels_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch_idx, (images, val_labels_batch) in enumerate(val_loader):
                images, val_labels_batch = images.to(device), val_labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, val_labels_batch)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += val_labels_batch.size(0)
                correct += (predicted == val_labels_batch).sum().item()
                all_labels.extend(val_labels_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # Identify misclassified neurons
                misclassified_indices = (predicted != val_labels_batch).nonzero(as_tuple=True)[0]
                for idx in misclassified_indices:
                    global_idx = batch_idx * val_loader.batch_size + idx.item()
                    misclassified_neurons.append({
                        'image_name': val_loader.dataset.image_paths[global_idx].split('/')[-1],
                        'original_file_name': image_name_to_file_name[val_loader.dataset.image_paths[global_idx].split('/')[-1]],
                        'true_label': val_labels_batch[idx].item(),
                        'predicted_label': predicted[idx].item()
                    })

                # Identify correctly classified neurons
                correctly_classified_indices = (predicted == val_labels_batch).nonzero(as_tuple=True)[0]
                for idx in correctly_classified_indices:
                    global_idx = batch_idx * val_loader.batch_size + idx.item()
                    correct_classified_neurons.append({
                        'image_name': val_loader.dataset.image_paths[global_idx].split('/')[-1],
                        'original_file_name': image_name_to_file_name[val_loader.dataset.image_paths[global_idx].split('/')[-1]],
                        'true_label': val_labels_batch[idx].item(),
                        'predicted_label': predicted[idx].item()
                    })

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Calculate precision, recall, F1 score, and ROC AUC
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Append fold metrics to list for writing to file
        metrics_per_fold.append({
            'Fold': fold + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': fscore,
            'ROC AUC': roc_auc
        })

        # Store metrics for each fold to compute averages later
        avg_metrics['train_losses'].append(train_loss)
        avg_metrics['train_accuracies'].append(train_acc)
        avg_metrics['val_losses'].append(val_loss)
        avg_metrics['val_accuracies'].append(val_acc)
        avg_metrics['precisions'].append(precision)
        avg_metrics['recalls'].append(recall)
        avg_metrics['fscores'].append(fscore)
        avg_metrics['roc_aucs'].append(roc_auc)
        avg_metrics['conf_matrices'].append(conf_matrix)

    # Save misclassified and correctly classified neurons for validation
    misclassified_file = f'misclassified_neurons_fold_{fold + 1}.csv'
    with open(misclassified_file, 'w') as f:
        f.write("image_name,original_file_name,true_label,predicted_label\n")
        for entry in misclassified_neurons:
            f.write(f"{entry['image_name']},{entry['original_file_name']},{entry['true_label']},{entry['predicted_label']}\n")
    print(f"Misclassified neurons for fold {fold + 1} saved to {misclassified_file}")

    correctly_classified_file = f'correctly_classified_neurons_fold_{fold + 1}.csv'
    with open(correctly_classified_file, 'w') as f:
        f.write("image_name,original_file_name,true_label,predicted_label\n")
        for entry in correct_classified_neurons:
            f.write(f"{entry['image_name']},{entry['original_file_name']},{entry['true_label']},{entry['predicted_label']}\n")
    print(f"Correctly classified neurons for fold {fold + 1} saved to {correctly_classified_file}")

    # Apply Grad-CAM for original images
    gradcam_save_dir = f'gradcam_results_fold_{fold + 1}'
    os.makedirs(gradcam_save_dir, exist_ok=True)

    target_layer = model.features[40]  # Correct layer for VGG16 with batch norm
    gradcam = GradCAM(model, target_layer)

    for image_path in val_image_paths:
        original_name = os.path.basename(image_path).split('.')[0]
        overlay, heatmap, _ = apply_gradcam(image_path, model, gradcam, transform, original_name)
        save_comparison_image(image_path, overlay, heatmap, original_name, gradcam_save_dir)

    # Apply masking and re-run Grad-CAM
    for mask_type in ['soma', 'dendrites']:
        mask_save_dir = f'gradcam_results_{mask_type}_fold_{fold + 1}'
        os.makedirs(mask_save_dir, exist_ok=True)

        for image_path in val_image_paths:
            original_name = os.path.basename(image_path).split('.')[0]
            masked_image = apply_mask(Image.open(image_path).convert('RGB'), mask_type)
            masked_image_path = os.path.join(mask_save_dir, f"{original_name}_{mask_type}.png")
            masked_image.save(masked_image_path)

            overlay, heatmap, _ = apply_gradcam(masked_image_path, model, gradcam, transform, original_name)
            save_comparison_image(masked_image_path, overlay, heatmap, original_name, mask_save_dir)

# Save fold metrics to a CSV
metrics_df = pd.DataFrame(metrics_per_fold)
metrics_df.to_csv('metrics_per_fold.csv', index=False)

# Compute and save the average metrics across folds
avg_metrics_df = pd.DataFrame({
    'Metric': ['Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Average': [
        np.mean(avg_metrics['train_losses']),
        np.mean(avg_metrics['train_accuracies']),
        np.mean(avg_metrics['val_losses']),
        np.mean(avg_metrics['val_accuracies']),
        np.mean(avg_metrics['precisions']),
        np.mean(avg_metrics['recalls']),
        np.mean(avg_metrics['fscores']),
        np.mean(avg_metrics['roc_aucs'])
    ]
})
avg_metrics_df.to_csv('average_metrics.csv', index=False)

# Average confusion matrix
avg_conf_matrix = np.mean(avg_metrics['conf_matrices'], axis=0).astype(int)
pd.DataFrame(avg_conf_matrix).to_csv('average_confusion_matrix.csv', index=False)

# Testing function with fixed Grad-CAM and metrics
def test_with_new_neurons(model, transform, device, labels, image_paths):
    new_ven_path = r'/mnt/c/Users/lrm22005/OneDrive - University of Connecticut/Research/Von_Economo_neurons/NewVENs/withSOMA'
    new_ven_image_paths = [os.path.join(new_ven_path, f) for f in os.listdir(new_ven_path) if f.endswith('.png')]
    new_ven_labels = [1] * len(new_ven_image_paths)  # All new neurons are von Economo

    # Randomly select an equal number of Pyramidal neurons from the original dataset for testing
    pyramidal_indices = [i for i, label in enumerate(labels) if label == 0]  # Use 'labels' passed as an argument
    np.random.shuffle(pyramidal_indices)
    selected_pyramidal_indices = pyramidal_indices[:len(new_ven_image_paths)]

    pyramidal_image_paths = [image_paths[i] for i in selected_pyramidal_indices]
    pyramidal_labels = [labels[i] for i in selected_pyramidal_indices]

    # Combine new Von Economo neurons and selected Pyramidal neurons
    test_image_paths = new_ven_image_paths + pyramidal_image_paths
    test_labels = new_ven_labels + pyramidal_labels

    test_dataset = NeuronImageDataset(test_image_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()  # Keep evaluation mode for regular predictions
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    misclassified_neurons = []  # To store misclassified neurons
    correct_classified_neurons = []  # To store correctly classified neurons

    # Directory to save Grad-CAM results for the testing phase
    gradcam_save_dir = 'gradcam_results_test'
    os.makedirs(gradcam_save_dir, exist_ok=True)

    # Apply Grad-CAM to testing
    target_layer = model.features[40]  # Correct layer for VGG16 with batch norm
    gradcam = GradCAM(model, target_layer)

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Regular forward pass for predictions
        with torch.no_grad():  # No gradients needed here
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Identify misclassified neurons
            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_indices:
                global_idx = batch_idx * test_loader.batch_size + idx.item()
                misclassified_neurons.append({
                    'image_name': test_loader.dataset.image_paths[global_idx].split('/')[-1],
                    'true_label': labels[idx].item(),
                    'predicted_label': predicted[idx].item()
                })

            # Identify correctly classified neurons
            correctly_classified_indices = (predicted == labels).nonzero(as_tuple=True)[0]
            for idx in correctly_classified_indices:
                global_idx = batch_idx * test_loader.batch_size + idx.item()
                correct_classified_neurons.append({
                    'image_name': test_loader.dataset.image_paths[global_idx].split('/')[-1],
                    'true_label': labels[idx].item(),
                    'predicted_label': predicted[idx].item()
                })

        # Grad-CAM requires gradients, so we manually enable gradient tracking
        for image_path in test_image_paths:
            original_name = os.path.basename(image_path).split('.')[0]

            # Enable gradient tracking for Grad-CAM
            with torch.set_grad_enabled(True):
                overlay, heatmap, _ = apply_gradcam(image_path, model, gradcam, transform, original_name)
                save_comparison_image(image_path, overlay, heatmap, original_name, gradcam_save_dir)

        # Apply masking and re-run Grad-CAM for test images
        for mask_type in ['soma', 'dendrites']:
            mask_save_dir = f'gradcam_results_{mask_type}_test'
            os.makedirs(mask_save_dir, exist_ok=True)

            for image_path in test_image_paths:
                original_name = os.path.basename(image_path).split('.')[0]
                masked_image = apply_mask(Image.open(image_path).convert('RGB'), mask_type)
                masked_image_path = os.path.join(mask_save_dir, f"{original_name}_{mask_type}.png")
                masked_image.save(masked_image_path)

                # Enable gradient tracking for Grad-CAM on masked images
                with torch.set_grad_enabled(True):
                    overlay, heatmap, _ = apply_gradcam(masked_image_path, model, gradcam, transform, original_name)
                    save_comparison_image(masked_image_path, overlay, heatmap, original_name, mask_save_dir)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Calculate metrics and save results
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Pyramidal', 'von Economo'])
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)

    # Save test metrics to a file
    test_metrics = {
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': fscore,
        'ROC AUC': roc_auc
    }
    pd.DataFrame([test_metrics]).to_csv('test_metrics.csv', index=False)

    # Save confusion matrix for testing
    pd.DataFrame(conf_matrix).to_csv('confusion_matrix_test.csv', index=False)

    # Save list of test neurons and their predictions
    test_neurons_file = 'testing_neurons.csv'
    pd.DataFrame({
        'image_name': test_image_paths,
        'true_label': test_labels,
        'predicted_label': all_preds
    }).to_csv(test_neurons_file, index=False)
    print(f"Testing neurons saved to {test_neurons_file}")

    # Plot ROC curve for testing
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Test Set')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_test.png')
    plt.show()
# Test the model with new neurons
test_with_new_neurons(model, transform, device, labels, image_paths)