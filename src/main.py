import torch
import torch.optim as optim
from network import SiameseThreeDim
from loss_functions import ConstractiveLoss
from loader import create_subject_pairs, transform_subjects, create_loaders
import os
from visualizations import single_layer_similar_heatmap_visual, multiple_layer_similar_heatmap_visiual
from distance_measures import various_distance
import cv2

def predict(siamese_net, test_loader, threshold=0.3):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for index, subject in enumerate(test_loader):
            input1 = subject['t1']['data'].float().to(device)
            input2 = subject['t2']['data'].float().to(device)
            label = subject['label'].to(device)
            output1, output2 = siamese_net(input1, input2)
            distance1 = various_distance(output1, output2, 'l2')  # Compute the distance euclidean
            distance = torch.dist(output1, output2, p=2)
            # print(f"Distance: {distance}")
            #similarity_score = 1 - distance.item()  # Convert distance to similarity score
            prediction = distance < threshold  # Determine if the pair is similar based on the threshold
            if prediction:
                print("The pair is similar with a distance of:", distance, " label:", label)
            else:
                print("The pair is dissimilar with a distance of:", distance, " label:", label)

            # Visualize the similarity heatmap
            heatmap = multiple_layer_similar_heatmap_visiual(output1, output2, 'l2')
            # Save the heatmap
            save_path = f'./data/heatmaps/threedim/{index}.jpg'
            cv2.imwrite(save_path, heatmap)
def train(siamese_net, optimizer, criterion, train_loader, val_loader, epochs=100, patience=3, 
          save_dir='models', model_name='masked.pth', device=torch.device('cuda')):
    siamese_net.to(device)
    print(f"Number of samples in training set: {len(train_loader)}")
    print(f"Number of samples in validation set: {len(val_loader)}")
    
    print("\nStarting training...")
    best_loss = float('inf')
    consecutive_no_improvement = 0
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        for index, subject in enumerate(train_loader):
            print(index)
            print(subject['t1']['data'].shape)
            print(subject['t1']['data'].shape)
            input1 = subject['t1']['data'].float().to(device)
            input2 = subject['t2']['data'].float().to(device)
            siamese_net.train()  # switch to training mode
            label = subject['label'].to(device)
            output1, output2 = siamese_net(input1, input2)
            loss = criterion(output1, output2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        # Validation loop
        siamese_net.eval()  # switch to evaluation mode
        with torch.no_grad():
            for index, subject in enumerate(val_loader):
                input1 = subject['t1']['data'].float().to(device)
                input2 = subject['t2']['data'].float().to(device)
                output1, output2 = siamese_net(input1, input2)
                label = subject['label'].to(device)
                loss = criterion(output1, output2, label)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    siamese3Dnet = SiameseThreeDim()
    subjects_raw= create_subject_pairs(root= './data/processed/preop/BTC-preop', id=['t1_ants_aligned.nii.gz'])
    subjects = transform_subjects(subjects_raw)
    train_loader_t1, val_loader_t1, test_loader_t1 = create_loaders(subjects, split=(0.6, 0.2, 0.2))
    # for index, subject in enumerate(train_loader_t1):
    #     print(index)
    save_dir = './models'
    if os.path.exists(os.path.join(save_dir, '3d.pth')):
        siamese3Dnet.load_state_dict(torch.load(os.path.join(save_dir, '3d.pth')))
        print('Loaded the best model')
    else:
        criterion = ConstractiveLoss(margin=0.0)
        optimizer = optim.Adam(siamese3Dnet.parameters(), lr=0.001)
        # Train the Siamese network
        train(siamese3Dnet, optimizer, criterion, train_loader=train_loader_t1, val_loader=val_loader_t1,
              epochs=50, patience=5, save_dir='./models', model_name='3d.pth', device=device)
    
    
    predict(siamese3Dnet, test_loader_t1, 0.014)
    # heatmap = single_layer_similar_heatmap_visual(output1,output2, 'l2')
    # merge_images(img1_set[0][0].numpy(), img2_set[0][0].numpy(), heatmap, f'./data/heatmaps/raw/unmasked/{patient_id}.jpg')