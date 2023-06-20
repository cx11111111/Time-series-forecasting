from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from models.LSTM import *
from models.GRU import *
from models.Transformer import *

if __name__ == "__main__":
    x = 10
    test_MSE, test_RMSE, test_MAE, test_R2, = [], [], [], []
    for i in range(x):
        # Load the dataset
        dataset, scaler = load_dataset(config.dataset_path, i,show_data=True)
        # Prepare the dataset for training/testing
        subsequences = extract_subsequences(dataset['scaled'], dataset['column_names'], lag=config.lag)
        # Split the dataset into train/validation/test set
        train_loader, val_set, test_set = train_val_test_split(subsequences)

        # # Train the network
        if config.mode == "train":
            # Create new instance of the RNN
            #net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1], hidden_dim=config.hidden_dim,num_layers=config.num_layers)
            net = GRUModel(input_dim=train_loader.dataset[0][0].shape[-1], hidden_dim=config.hidden_dim,num_layers=config.num_layers)
            #net = TransformerModel(n_encoder_inputs=train_loader.dataset[0][0].shape[-1],n_decoder_inputs=config.output_size,num_heads=config.num_heads,
                                #Sequence_length=config.lag,d_model=config.hidden_dim,dropout=config.dropout,num_layer=config.num_layers)
            net.to(device)
        # Train the model
            train_loop(net, config.epochs, config.lr, config.wd, train_loader, val_set,i, debug=True)
        else:
            # Create new instance of the RNN using default values
            net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1],
                                hidden_dim=parser.get_default('hidden_dim'),
                                num_layers=parser.get_default('num_layers')).to(device)
            # Load pretrained weights
            net.load_state_dict(torch.load(config.pretrained_path, map_location=device))

        # Display the prediction next to the target output values
        test_MSE,test_RMSE,test_MAE,test_R2=show_evaluation(net, test_set,subsequences, scaler,i,test_MSE,test_RMSE,test_MAE,test_R2, debug=True)
    df = pd.DataFrame({
        'Test MSE': test_MSE,
        'Test RMSE': test_RMSE,
        'Test MAE': test_MAE,
        'Test R2': test_R2
    })
    df.to_csv('C:/Users/cx/Desktop/zhibiao1(144).csv', index=False)
