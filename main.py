from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_and_evaluate_model
from src.predict import predict_and_save

def main():
    # Load Data
    train_data, test_data = load_data('data/1.Train.csv', 'data/1.Test.csv')
    
    # Preprocess Data
    train_data, test_data = preprocess_data(train_data, test_data)
    
    # Prepare Features and Target
    X = train_data.drop('Item_Outlet_Sales', axis=1)
    y = train_data['Item_Outlet_Sales']
    
    # Train Model
    model = train_and_evaluate_model(X, y)
    
    # Make Predictions and Save
    predict_and_save(model, test_data, 'BIG_MART_SALES.csv')

if __name__ == "__main__":
    main()
