import torch
print(torch.cuda.is_available()) # should be True
# t = torch.rand(10, 10).cuda()
# print(t.device) # should be CUDA]
import torch.optim as optim
import torchvision.transforms as transforms
from network import SiameseThreeDim
from loss_functions import ConstractiveLoss
from loader import create_subject_pairs, transform_subjects, create_loaders
from PIL import Image
import os
import nibabel as nib
import matplotlib.pyplot as plt
import torchio as tio




def train(siamese_net, optimizer, criterion, train_loader, val_loader, epochs=100, patience=3, 
          save_dir='models', model_name='masked.pth'):
    print(f"Number of samples in training set: {len(train_loader)}")
    print(f"Number of samples in validation set: {len(val_loader)}")
    
    print("\nStarting training...")
    best_loss = float('inf')
    consecutive_no_improvement = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        for index, subject in enumerate(train_loader):
            siamese_net.train()  # switch to training mode
            output1, output2 = siamese_net(subject['t1']['data'].float(), subject['t2']['data'].float())
            loss = criterion(output1, output2, subject['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        # Validation loop
        siamese_net.eval()  # switch to evaluation mode
        with torch.no_grad():
            for index, subject in enumerate(val_loader):
                output1, output2 = siamese_net(subject['t1']['data'].float(), subject['t2']['data'].float())
                loss = criterion(output1, output2, subject['label'])
                epoch_val_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Check for improvement in validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            consecutive_no_improvement = 0
            # Save the best model
            save_path = os.path.join(save_dir, model_name)
            torch.save(siamese_net.state_dict(), save_path)
            print(f'Saved best model to {save_path}')
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1} as no improvement for {patience} consecutive epochs.')
                break


if __name__ == "__main__":
    print("Test")
    siamese3Dnet = SiameseThreeDim()
    subjects_raw= create_subject_pairs(root= './data/processed/preop/BTC-preop', id=['t1_ants_aligned.nii.gz'])
    subjects = transform_subjects(subjects_raw)
    train_loader_t1, val_loader_t1, test_loader_t1 = create_loaders(subjects, split=(0.6, 0.2, 0.2))
    # siamese3Dnet = siamese3Dnet.cuda()  # Move the network to GPU
    save_dir = './models'
    if os.path.exists(os.path.join(save_dir, '3d.pth')):
        siamese3Dnet.load_state_dict(torch.load(os.path.join(save_dir, '3d.pth')))
        print('Loaded the best model')
    else:
        criterion = ConstractiveLoss(margin=0.3)
        optimizer = optim.Adam(siamese3Dnet.parameters(), lr=0.001)
        # Train the Siamese network
        train(siamese3Dnet,  optimizer, criterion, train_loader=train_loader_t1, val_loader=val_loader_t1,
            epochs=10, patience=5, save_dir='./models', model_name='3d.pth')