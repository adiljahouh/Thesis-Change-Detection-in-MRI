import torch
import torch.optim as optim
from network import SiameseThreeDim, SiameseVGG3D
from loss_functions import ConstractiveLoss
from loader import create_subject_pairs, transform_subjects, create_loaders_with_index, create_loaders_with_split
import os
from visualizations import multiple_layer_similar_heatmap_visiual, generate_roc_curve
import argparse
import cv2
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
## using leave-one-out cross validation because our data is small
## change loss function to incorporate multiple layer losses
def predict(siamese_net, test_loader, threshold=0.3):
    siamese_net.to(device)
    siamese_net.eval()  # Set the model to evaluation mode
    distances = []
    labels = []
    with torch.no_grad():
        for index, subject in enumerate(test_loader):
            input1 = subject['t1']['data'].float().to(device)
            input2 = subject['t2']['data'].float().to(device)
            label = subject['label'].to(device)
            output1, output2 = siamese_net(input1, input2)
            distance = ConstractiveLoss(output1, output2, p=2)
            distances.append(distance)
            labels.append(label)
            # print(f"Distance: {distance}")
            #similarity_score = 1 - distance.item()  # Convert distance to similarity score
            prediction = distance < threshold  # Determine if the pair is similar based on the threshold
            if prediction:
                print("The pair is similar with a distance of:", distance.item(), " label:", label.item())
            else:
                print("The pair is dissimilar with a distance of:", distance.item(), " label:", label.item())

            # Visualize the similarity heatmap
            heatmap = multiple_layer_similar_heatmap_visiual(output1, output2, 'l2')
            # Save the heatmap
            save_dir = os.path.join(os.getcwd(), f'./data/heatmaps/threedim/{subject["name"][0]}')
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            for i, img in enumerate(heatmap):
                save_path = f'./data/heatmaps/threedim/{subject["name"][0]}/{i}.jpg'
                cv2.imwrite(os.path.join(os.getcwd(), save_path), img)
    return distances, labels

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
    parser = argparse.ArgumentParser(description="Siamese Network Operations")

    parser.add_argument('--model', type=str, choices=['custom', 'vgg16'],
                             help='Type of model architecture to use (custom or VGG16-based).', 
                             required=True)
    parser.add_argument("--dist_flag", type=str, choices=['l2', 'l1', 'cos'], required=True, help=
                        "Distance flag to use for the loss function (l2, l1, or cos)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin of the constractive loss")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    save_dir = f'./models/{args.model}'
    loo = LeaveOneOut()
    subjects_raw= create_subject_pairs(root= './data/processed/preop/BTC-preop', id=['t1_ants_aligned.nii.gz'])
    subjects = transform_subjects(subjects_raw)
    print(f"Number of subjects: {len(subjects)}")
    if args.model == 'custom':
        model_type = SiameseThreeDim()
    elif args.model == 'vgg16':
        model_type = SiameseVGG3D()
    criterion = ConstractiveLoss(margin=args.margin, dist_flag=args.dist_flag)
    optimizer = optim.Adam(model_type.parameters(), lr=args.lr)
                # Train the network
    for i, (train_index, test_index) in enumerate(loo.split(subjects)):
        train_val_loader, test_loader = create_loaders_with_index(subjects, train_index, test_index)
        train_loader, val_loader = create_loaders_with_split(train_val_loader.dataset, 
                                                             split=(0.8, 0.2), generator=
                                                             torch.random.manual_seed(42))
        train(model_type, optimizer, criterion, train_loader=train_loader, val_loader=val_loader, 
            epochs=args.epochs, patience=args.patience, 
            save_dir=save_dir, model_name=f'{args.model}_{args.dist_flag}_'\
            f'lr-{args.lr}_marg-{args.margin}.pth', device=device)
        
        ##TODO: fix validation method
        distances, labels = predict(model_type, test_loader, 7)
        #thresholds = generate_roc_curve(distances, labels, f"./models/{args.model}")


