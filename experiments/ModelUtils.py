
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def getFcModel(
        input_size, 
        output_size, 
        num_layers, 
        neurons_hidden_layer, 
        activation, 
        dropout ):
    """Creates a Fully Connected model with Pytorch 

        Args:
            input_size: size of network input
            output_size: size of network output
            num_layers: number of layers
            neurons_hidden_layer: neurons on each layer

        Returns: 
            - The fully connected model
    """

    class FcModel(nn.Module):

        def __init__(self, activation, dropout):
            super(FcModel, self).__init__()

            self.dropout = dropout
            self.dropout1 = nn.Dropout(0.05)
            self.dropout2 = nn.Dropout(0.50)
            self.dropout3 = nn.Dropout(0.50)
            self.dropout4 = nn.Dropout(0.50)
            self.dropout5 = nn.Dropout(0.50)

            if activation=='relu':
                self.activation_f = F.relu
            
            elif activation == 'sigmoid':
                self.activation_f = torch.sigmoid

            else:
                raise Exception("Activation not recognized")

            # 1 Layer
            if num_layers == 1:
                self.fc1 = nn.Linear(input_size, output_size) 

            # 2 Layers
            if num_layers == 2:
                self.fc1 = nn.Linear(input_size, neurons_hidden_layer) 
                self.fc2 = nn.Linear(neurons_hidden_layer, output_size)

            # 3 Layers
            if num_layers == 3:
                self.fc1 = nn.Linear(input_size, neurons_hidden_layer) 
                self.fc2 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc3 = nn.Linear(neurons_hidden_layer, output_size)
            
            # 4 Layers
            if num_layers == 4:
                self.fc1 = nn.Linear(input_size, neurons_hidden_layer) 
                self.fc2 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc3 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc4 = nn.Linear(neurons_hidden_layer, output_size)

            # 5 Layers
            if num_layers == 5:
                self.fc1 = nn.Linear(input_size, neurons_hidden_layer) 
                self.fc2 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc3 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc4 = nn.Linear(neurons_hidden_layer, neurons_hidden_layer) 
                self.fc5 = nn.Linear(neurons_hidden_layer, output_size)

        def forward(self, x):
            x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
            
            if num_layers == 1:

                if self.dropout:
                    x = self.dropout1(x)

                x = self.fc1(x)

            elif num_layers == 2:
                if self.dropout:
                    x = self.dropout1(x)

                x = self.fc1(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout2(x)

                x = self.fc2(x)

            elif num_layers == 3:
                if self.dropout:
                    x = self.dropout1(x)

                x = self.fc1(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout2(x)

                x = self.fc2(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout3(x)

                x = self.fc3(x)

            elif num_layers == 4:
                if self.dropout:
                    x = self.dropout1(x)

                x = self.fc1(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout2(x)

                x = self.fc2(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout3(x)

                x = self.fc3(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout4(x)

                x = self.fc4(x)

            elif num_layers == 5:
                if self.dropout:
                    x = self.dropout1(x)

                x = self.fc1(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout2(x)

                x = self.fc2(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout3(x)

                x = self.fc3(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout4(x)

                x = self.fc4(x)
                x = self.activation_f(x)

                if self.dropout:
                    x = self.dropout5(x)

                x = self.fc5(x)

            else: 
                raise Exception("Number of hidden layers out of range")

            return x

    model = FcModel(activation, dropout)

    # Initialize model weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    return model

def train_TransferLearning_Simultaneous_Backprop_PC(
    epochs, 
    num_classes, 
    train_generator, 
    valid_generator, 
    model,
    feature_extractor, 
    criterion, 
    optimizer, 
    device,
    print_every_n_batches,
    pc_model = None):

    """ Trains the FC layer and a Predictive Coding network simultaneously, given an image dataset generator and a feature extractor

        Args:
            epochs
            num_classes
            train_generator
            valid_generator 
            model: The fully connected model from Pytorch
            feature_extractor: The feature extractor model
            criterion: Pytorch loss calculator for FC model
            optimizer: Pytorch optimizer for FC model
            device: The CPU or GPU device
            print_every_n_batches: how often to print training accuracy

        Returns: 
            - A dictionary with the evaluation metrics of the training session
    """
    
    # Return metrics after training
    metrics = {
        'backprop_train_acc': [],
        'backprop_val_acc': [],
        'pc_train_acc': [],
        'pc_val_acc': []
    }

    for epoch in range(epochs):
        running_loss = 0.0
        prediction_list = []
        prediction_list_pc = []
        labels_list = []

        print(f'\nEpoch: {epoch+1}')

        # Activate dropouts, batch norm...
        model.train()
        feature_extractor.train()
        
        for i, (data, labels) in enumerate(train_generator):

            # Get samples
            data = data.to(device)
            labels = labels.to(device)

            # Zero model gradiants
            model.zero_grad() 

            # Compute features
            features = feature_extractor(data)

            # Comput model output
            prediction = model(features)

            # Calculate loss and gradiants
            loss = criterion(prediction, labels)
            loss.backward()

            # Apply gradients
            optimizer.step()

            # Get running loss
            running_loss += loss.item()

            # Store predictions
            max_index = prediction.max(dim = 1)[1]
            prediction_list.extend(list(max_index.to('cpu').numpy()))
            labels_list.extend(labels.to('cpu').numpy())

            # Predictive Coding training
            if pc_model:
                labels_one_hot = F.one_hot(labels, num_classes=num_classes)
                prediction_pc = pc_model.single_batch_pass(features, labels_one_hot)
                prediction_list_pc.extend(list(torch.argmax(prediction_pc, dim=0).to('cpu').numpy()))
                
            # Calculate partial training accuracy
            if i % print_every_n_batches == print_every_n_batches-1:    # print every N mini-batches

                # Training metrics 
                acc_metric = np.equal(prediction_list, labels_list).sum()*1.0/len(prediction_list)
                acc_metric_pc = np.nan

                if pc_model:
                    acc_metric_pc = np.equal(prediction_list_pc, labels_list).sum()*1.0/len(prediction_list_pc)

                print('batch num: %5d, (backprop) acc: %.3f | (pc) acc: %.3f' % 
                    (i + 1, acc_metric, acc_metric_pc))
        
        # Finished epoch

        # Calculate validation accuracy and train accuracy for epoch

        acc_metric = np.equal(prediction_list, labels_list).sum()*1.0/len(prediction_list)
        acc_metric_pc = np.nan
        if pc_model:
            acc_metric_pc = np.equal(prediction_list_pc, labels_list).sum()*1.0/len(prediction_list_pc)
        
        prediction_list_valid = []
        prediction_list_pc_valid = []
        labels_list_valid = []
        
        #   Disable dropouts: model.eval()
        model.eval()
        feature_extractor.eval()

        for data, labels in valid_generator:
            # Get samples
            data = data.to(device)
            labels = labels.to(device)

            # Compute features
            features = feature_extractor(data)

            # Comput model output
            prediction = model(features)

            # Calculate loss
            loss = criterion(prediction, labels)

            # Store predictions
            max_index = prediction.max(dim = 1)[1]
            prediction_list_valid.extend(list(max_index.to('cpu').numpy()))
            labels_list_valid.extend(labels.to('cpu').numpy())

            if pc_model:
                prediction_pc = pc_model.batch_inference(features)
                prediction_list_pc_valid.extend(list(torch.argmax(prediction_pc, dim=0).to('cpu').numpy()))
            
        # Validation metrics 
        valid_accuracy = np.equal(prediction_list_valid, labels_list_valid).sum()*1.0/len(prediction_list_valid)
        valid_accuracy_pc = np.nan

        if pc_model:
            valid_accuracy_pc = np.equal(prediction_list_pc_valid, labels_list_valid).sum()*1.0/len(prediction_list_pc_valid)

        # Print Loss and Accuracy 
        print('Epoch: %d, (backprop) acc: %.3f, val acc: %.3f | (pc) acc: %.3f, val acc: %.3f' % 
            (epoch + 1, acc_metric, valid_accuracy, acc_metric_pc, valid_accuracy_pc))
        
        running_loss = 0.0
        prediction_list = []
        labels_list = []

        # Store metrics for epoch
        metrics['backprop_train_acc'].append(acc_metric)
        metrics['backprop_val_acc'].append(valid_accuracy)

        if pc_model:
            metrics['pc_train_acc'].append(acc_metric_pc)
            metrics['pc_val_acc'].append(valid_accuracy_pc)

    return metrics

def printMetrics(metrics):

    """ Prints neural network training metrics

        Args:
            metrics: Output of `train_TransferLearning_Simultaneous_Backprop_PC()`

    """

    print("------------------------------------------------")
    print("End of training session\n")

    print("backprop_train_acc=", end="", flush=True)
    print(metrics['backprop_train_acc'])

    print("backprop_val_acc=", end="", flush=True)
    print(metrics['backprop_val_acc'])

    print("pc_train_acc=", end="", flush=True)
    print(metrics['pc_train_acc'])

    print("pc_val_acc=", end="", flush=True)
    print(metrics['pc_val_acc'])

def getPcModelArchitecture(input_size, output_size, num_layers, neurons_hidden_layer):
    neurons = [input_size]
    for i in range(num_layers-1):
        neurons.append(neurons_hidden_layer)

    neurons.append(output_size)

    return neurons
    


