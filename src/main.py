import torch
import torch.optim as optim
from network import SiameseThreeDim, SiameseVGG3D
from loss_functions import ConstractiveLoss
from loader import create_subject_pairs, transform_subjects
import os
from visualizations import multiple_layer_similar_heatmap_visiual, generate_roc_curve
import argparse
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import random_split, DataLoader, Subset
import torch.nn.functional as F
## using leave-one-out cross validation because our data is small
## using stratified kfold for cross validation because LOO will overfit and 
## Requested server, and data storage
## installed python, conda and setup pytorch on server
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
            distance = torch.dist(output1,output2,p=2)
            #distance = torch.dist(output1, output2, p=2)
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
            # heatmap = multiple_layer_similar_heatmap_visiual(output1, output2, 'l2')
            # # Save the heatmap
            # save_dir = os.path.join(os.getcwd(), f'./data/heatmaps/threedim/{subject["name"][0]}')
            # os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            # for i, img in enumerate(heatmap):
            #     save_path = f'./data/heatmaps/threedim/{subject["name"][0]}/{i}.jpg'
            #     cv2.imwrite(os.path.join(os.getcwd(), save_path), img)
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
    return best_loss           

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
    subjects_raw= create_subject_pairs(root= './data/processed/preop/BTC-preop', 
                                       id=['t1_ants_aligned.nii.gz'])
    subjects = transform_subjects(subjects_raw)
    
    # Train the network using kFold cross validation
    validation_accuracy = []
    fold_data = {}
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for i, (train_index, test_val_index) in enumerate(skf.split(subjects, 
                                                            [subject.label for subject in subjects])):
        subjects_train = Subset(subjects, train_index)
        subjects_val_test = Subset(subjects, test_val_index)
        subjects_val, subjects_test = train_test_split(subjects_val_test, test_size=0.5, random_state=42, stratify=[subject.label for subject in subjects_val_test])
        print(f"Number of test subjects: {len(subjects_test)}")

        # we are remembering the test set which we split off for later cv
        fold_data[i] = (subjects_train, subjects_val, subjects_test)
        if args.model == 'custom':
            model_type = SiameseThreeDim()
        elif args.model == 'vgg16':
            model_type = SiameseVGG3D()
        criterion = ConstractiveLoss(margin=args.margin, dist_flag=args.dist_flag)
        optimizer = optim.Adam(model_type.parameters(), lr=args.lr)

        ## using validation split to avoid overfitting
        train_loader = DataLoader(subjects_train, batch_size=1, shuffle=False)
        val_loader = DataLoader(subjects_test, batch_size=1, shuffle=False)
        best_loss = train(model_type, optimizer, criterion, train_loader=train_loader, val_loader=val_loader, 
            epochs=args.epochs, patience=args.patience, 
            save_dir=save_dir, model_name=f'{args.model}_{args.dist_flag}_'\
            f'lr-{args.lr}_marg-{args.margin}_fold-{i}.pth', device=device)
        validation_accuracy.append(best_loss)
        print(f"Appending validation accuracy for fold {i}: {best_loss}")

    print(f"Average validation accuracy for all folds: {sum(validation_accuracy)/len(validation_accuracy)}")
    # Test the model on the test set
    index_best_model = validation_accuracy.index(min(validation_accuracy))
    best_model = f'{args.model}_{args.dist_flag}_lr-{args.lr}_marg-{args.margin}_fold-{index_best_model}.pth'
    print(f"Using the best model: {best_model}")
    model_path = os.path.join(save_dir, best_model)

    model_type.load_state_dict(torch.load(model_path))
    merged_distances, merged_labels = [], []
    for i in range(4):
        subjects_train, subjects_val, subjects_test = fold_data[i]
        test_loader = DataLoader(subjects_test, batch_size=1, shuffle=False)
        distances, labels = predict(model_type, test_loader, 7)
        merged_distances.extend(distances)
        merged_labels.extend(labels)
    thresholds = generate_roc_curve(merged_distances, merged_labels, f"./models/{args.model}")



