import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from PIL import Image, ImageFile
from sklearn import metrics
import seaborn as sns
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.makedirs('/cs/student/projects1/2022/ngoyal/robotics/model_cache', exist_ok=True)
os.environ['TORCH_HOME'] = '/cs/student/projects1/2022/ngoyal/robotics/model_cache'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

class WheelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((os.path.join(class_dir, img_name), 
                                        self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(360),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=None,
            shear=(-15, 15)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'realistic_test': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def load_data(data_dir):
    image_datasets = {
        x: WheelDataset(os.path.join(data_dir, x), data_transforms[x]) 
        for x in ['train', 'val', 'realistic_test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
        for x in ['train', 'val', 'realistic_test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# def compare_hyperparameters(dataloaders, dataset_sizes, num_classes):
#     """Compare different hyperparameter settings"""
#     histories = {}
    
#     # Define hyperparameter configurations to try
#     configs = {
#         'baseline': {
#             'lr': 0.001,
#             'dropout': 0.5,
#             'label_smoothing': 0.1
#         },
#         'higher_lr': {
#             'lr': 0.005,
#             'dropout': 0.5,
#             'label_smoothing': 0.1
#         },
#         'lower_dropout': {
#             'lr': 0.001,
#             'dropout': 0.2,
#             'label_smoothing': 0.1
#         },
#         'no_smoothing': {
#             'lr': 0.001,
#             'dropout': 0.5,
#             'label_smoothing': 0.0
#         }
#     }
    
#     for config_name, params in configs.items():
#         print(f"\nTraining with {config_name} configuration...")
        
#         # Create model
#         model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
#         # Freeze parameters
#         for param in model.parameters():
#             param.requires_grad = False
#         for param in model.layer4.parameters():
#             param.requires_grad = True
        
#         # Modify FC layer with configured dropout
#         num_features = model.fc.in_features
#         model.fc = nn.Sequential(
#             nn.Dropout(params['dropout']),
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(params['dropout'] * 0.6),  # Slightly lower dropout for second layer
#             nn.Linear(256, num_classes)
#         )
        
#         model = model.to(device)
        
#         # Define training components with configured hyperparameters
#         criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
#         optimizer = optim.Adam(model.fc.parameters(), lr=params['lr'])
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
#         )
        
#         # Train
#         model, history = train_model(
#             model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
#             f"hp_{config_name}", num_epochs=30
#         )
        
#         histories[config_name] = history
    
#     # Plot comparison
#     plt.figure(figsize=(15, 6))
    
#     # Validation accuracy
#     plt.subplot(1, 2, 1)
#     for config_name, history in histories.items():
#         plt.plot(history['val_acc'], label=config_name)
#     plt.title('Validation Accuracy by Hyperparameters')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     # Training loss
#     plt.subplot(1, 2, 2)
#     for config_name, history in histories.items():
#         plt.plot(history['train_loss'], label=config_name)
#     plt.title('Training Loss by Hyperparameters')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig("hyperparameter_comparison.png")
#     plt.show()
    
#     return histories

# def compare_base_models(dataloaders, dataset_sizes, num_classes):
#     """Compare different pre-trained models for transfer learning"""
#     histories = {}
#     # Define different base models to try
#     base_models = {
#     'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
#     'mobilenet_v2': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
#     'efficientnet_b0': models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
#     }
#     for model_name, base_model in base_models.items():
#         print(f"\nTraining with {model_name} as base model...")
#         # Freeze base model parameters
#         for param in base_model.parameters():
#             param.requires_grad = False
#         # Modify classifier/FC layer based on model architecture
#         if model_name == 'resnet18':
#             num_features = base_model.fc.in_features
#             base_model.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#             )
#         elif model_name == 'mobilenet_v2':
#             num_features = base_model.classifier[1].in_features
#             base_model.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#             )
#         elif model_name == 'efficientnet_b0':
#             num_features = base_model.classifier[1].in_features
#             base_model.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
#         model = base_model.to(device)
#         optimizer = optim.Adam(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=0.001
#             )
#         criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
#             )

#         model, history = train_model(
#             model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
#             model_name, num_epochs=30
#             )
#         histories[model_name] = history
    
#     # Plot comparison
#     plt.figure(figsize=(15, 6))
    
#     # Validation accuracy
#     plt.subplot(1, 2, 1)
#     for model_name, history in histories.items():
#         plt.plot(history['val_acc'], label=model_name)
#     plt.title('Validation Accuracy by Base Model')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     # Training loss
#     plt.subplot(1, 2, 2)
#     for model_name, history in histories.items():
#         plt.plot(history['train_loss'], label=model_name)
#     plt.title('Training Loss by Base Model')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("base_model_comparison_classifier_only.png")
#     plt.show()
#     return histories

# def compare_unfrozen_layers(dataloaders, dataset_sizes, num_classes):
#     """Compare performance when unfreezing different ResNet18 layers"""
#     histories = {}
#     layer_configs = {
#         'fc_only': [],  # Freeze everything except FC
#         'layer4': ['layer4'],
#         'layer3': ['layer3'],
#         'layer3_4': ['layer3', 'layer4'],
#         'all_layers': ['layer1', 'layer2', 'layer3', 'layer4']
#     }
    
#     for config_name, layers_to_unfreeze in layer_configs.items():
#         print(f"\nTraining with {config_name} unfrozen...")
        
#         # Create model
#         model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
#         # Freeze all parameters by default
#         for param in model.parameters():
#             param.requires_grad = False
        
#         # Unfreeze specified layers
#         for layer_name in layers_to_unfreeze:
#             for param in getattr(model, layer_name).parameters():
#                 param.requires_grad = True
        
#         # Modify FC layer
#         num_features = model.fc.in_features
#         model.fc = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(num_features, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.2* 0.6),
#             nn.Linear(256, num_classes)
#         )
        
#         model = model.to(device)
        
#         # Define optimizer with different learning rates
#         params_to_update = [
#             {'params': model.fc.parameters(), 'lr': 0.005}
#         ]
        
#         # Add unfrozen layers with lower learning rate
#         for layer_name in layers_to_unfreeze:
#             params_to_update.append(
#                 {'params': getattr(model, layer_name).parameters(), 'lr': 0.0005}
#             )
        
#         optimizer = optim.Adam(params_to_update)
#         criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
#         )
        
#         # Train
#         model, history = train_model(
#             model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
#             f"resnet18_{config_name}", num_epochs=30
#         )
        
#         histories[config_name] = history
    
#     # Plot comparison
#     plt.figure(figsize=(15, 6))
    
#     # Validation accuracy
#     plt.subplot(1, 2, 1)
#     for config_name, history in histories.items():
#         plt.plot(history['val_acc'], label=config_name)
#     plt.title('Validation Accuracy by Unfrozen Layers')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     # Training loss
#     plt.subplot(1, 2, 2)
#     for config_name, history in histories.items():
#         plt.plot(history['train_loss'], label=config_name)
#     plt.title('Training Loss by Unfrozen Layers')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig("unfrozen_layers_comparison.png")
#     plt.show()
    
#     return histories

def plot_confidence_distribution(model, dataloader, save_path=None):
    """Plot distribution of confidence scores for correct and incorrect predictions"""
    model.eval()
    correct_conf = []
    incorrect_conf = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get confidence of predicted class
            confidence = probs[torch.arange(probs.size(0)), preds].cpu().numpy()
            
            # Split by correct/incorrect
            mask = (preds == labels).cpu().numpy()
            correct_conf.extend(confidence[mask])
            incorrect_conf.extend(confidence[~mask])
    
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 20)
    plt.hist(correct_conf, bins=bins, alpha=0.5, label='Correct predictions')
    plt.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect predictions')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_misclassified_examples(model, dataloader, class_names, num_examples=10, save_path=None):
    """Show examples of misclassified images with confidence scores"""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Find misclassified examples
            mask = preds != labels
            if torch.any(mask):
                for i in range(len(mask)):
                    if mask[i]:
                        misclassified.append({
                            'image': inputs[i].cpu(),
                            'true': labels[i].item(),
                            'pred': preds[i].item(),
                            'prob': probs[i, preds[i]].item()
                        })
            
            if len(misclassified) >= num_examples:
                break
    
    # Show the misclassified examples
    if misclassified:
        n_cols = min(5, num_examples)
        n_rows = (num_examples + n_cols - 1) // n_cols
        
        # Create figure with proper handling for different subplot configurations
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        
        # Handle the case when we have a single row and matplotlib doesn't return a 2D array
        if n_rows == 1:
            # For a single row, make sure axes is treated as a 1D array
            axes = [axes] if n_cols == 1 else axes
        else:
            # For multiple rows, flatten the 2D array
            axes = axes.flatten()
        
        for i, example in enumerate(misclassified[:num_examples]):
            img = example['image'].permute(1, 2, 0).numpy()
            # De-normalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {class_names[example['true']]}\n"
                             f"Pred: {class_names[example['pred']]}\n"
                             f"Conf: {example['prob']:.2f}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(misclassified), len(axes)):
            if i < len(axes):  # Make sure we don't go out of bounds
                axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    nn.init.xavier_uniform_(model.fc[1].weight)
    nn.init.zeros_(model.fc[1].bias)
    
    return model

def plot_confusion_matrix(model, dataloader, class_names, save_path=None):
    """Generate and plot confusion matrix"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return cm

# def show_class_examples(data_dir, save_path=None):
#     """Display examples of each class"""
#     class_dirs = sorted([d for d in os.listdir(os.path.join(data_dir, 'train')) 
#                        if os.path.isdir(os.path.join(data_dir, 'train', d))])
    
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     axes = axes.flatten()
    
#     for i, class_name in enumerate(class_dirs):
#         class_path = os.path.join(data_dir, 'train', class_name)
#         image_files = [f for f in os.listdir(class_path) 
#                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
#         if image_files:
#             # Get a random image from this class
#             img_path = os.path.join(class_path, random.choice(image_files))
#             img = Image.open(img_path).convert('RGB')
            
#             axes[i].imshow(img)
#             axes[i].set_title(f'Class: {class_name}')
#             axes[i].axis('off')
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# def show_augmentations(image_path, num_augmentations=5, save_path=None):
#     """Display multiple augmentations of the same image"""
#     image = Image.open(image_path).convert('RGB')
    
#     fig, axes = plt.subplots(2, num_augmentations, figsize=(15, 6))
#     axes[0, 0].imshow(image)
#     axes[0, 0].set_title('Original')
#     axes[0, 0].axis('off')
    
#     for i in range(1, num_augmentations):
#         aug_img = data_transforms['train'](image).permute(1, 2, 0).numpy()
#         aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#         aug_img = np.clip(aug_img, 0, 1)
        
#         axes[0, i].imshow(aug_img)
#         axes[0, i].set_title(f'Aug {i}')
#         axes[0, i].axis('off')
    
#     for i in range(num_augmentations):
#         tta_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
        
#         aug_img = tta_transform(image).permute(1, 2, 0).numpy()
#         aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#         aug_img = np.clip(aug_img, 0, 1)
        
#         axes[1, i].imshow(aug_img)
#         axes[1, i].set_title(f'VTA {i}')
#         axes[1, i].axis('off')
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, weights_f, num_epochs):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())

                scheduler.step(epoch_acc)
            
            if phase == 'val' and epoch_acc > best_acc - .5:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'{weights_f}.pth')

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    return model, history

def fine_tune_model(model):
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    optimizer_ft = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.0005},
        {'params': model.fc.parameters(), 'lr': 0.0005}
    ])
    
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='min', factor=0.5, patience=4, min_lr=1e-5
    )
    
    return model, optimizer_ft, scheduler_ft

def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def test_time_augmentation(model, image_path, class_names, num_augmentations=10):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    
    augmentation = data_transforms['realistic_test']
    predictions = []
    with torch.no_grad():
        for _ in range(num_augmentations):
            augmented_img = augmentation(image).unsqueeze(0).to(device)
            outputs = model(augmented_img)
            outputs = torch.softmax(outputs, dim=1)
            predictions.append(outputs.cpu().numpy())
    
    avg_prediction = np.mean(np.array(predictions), axis=0)
    predicted_class = np.argmax(avg_prediction)
    confidence = avg_prediction[0][predicted_class]
    
    return predicted_class, confidence, class_names[predicted_class]

def main(mode, weights_f):
    data_dir = 'wheel_dataset'
    
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    model = create_model(num_classes)
    model = model.to(device)
    if mode == "train":      
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5
        )
        
        print("Starting initial training phase...")
        model, history = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, weights_f, num_epochs=20
        )
        
        plot_training_history(history, f"FCL_Training_{weights_f}")
        
        print("\nStarting fine-tuning phase...")
        model.load_state_dict(torch.load(f'{weights_f}.pth'))
        model, optimizer_ft, scheduler_ft = fine_tune_model(model)
        model, history_ft = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer_ft, scheduler_ft, weights_f, num_epochs=40
        )
        
        plot_training_history(history_ft, f"Layer_4_Tuning_{weights_f}")
        
        print(f"Training complete. Model saved as '{weights_f}.pth'")
    elif mode == "test":
        model.load_state_dict(torch.load(f'{weights_f}.pth'))
        # for test_image_path in os.listdir(os.path.join(data_dir, "test")):
        #     _, confidence, class_name = test_time_augmentation(
        #         model, os.path.join(data_dir, "test", test_image_path), class_names
        #     )
        #     print(f"Predicted class: {class_name} with confidence: {confidence:.2f}")

        plot_confusion_matrix(model, dataloaders['val'], class_names, f"confusion_matrix_val_{weights_f}.png")
        show_misclassified_examples(model, dataloaders['realistic_test'], class_names, 5, f"misclassified_{weights_f}.png")
        plot_confidence_distribution(model, dataloaders['realistic_test'], f"confidence_distribution_{weights_f}.png")
        
        # show_class_examples("wheel_dataset", "dataset_examples.png")
        # compare_unfrozen_layers(dataloaders, dataset_sizes, num_classes)
        # compare_base_models(dataloaders, dataset_sizes, num_classes)
        # compare_hyperparameters(dataloaders, dataset_sizes, num_classes)
        # show_augmentations("wheel_dataset/val/BMW_M3_CSL/14.jpeg", 9, 'augmentations_BMW_M3_CSL.png')

        
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
                    prog='WheelClassifier',
                    description='Classifies 10 wheels',
                    epilog='give mode (train or test) and weights path for store/use')
    parser.add_argument('-m', '--mode', choices=['train', 'test'])
    parser.add_argument('-f', '--weights_f')
    args = parser.parse_args()
    main(args.mode, args.weights_f)