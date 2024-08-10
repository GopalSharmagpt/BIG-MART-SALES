import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_data, test_data):
    # Handle missing values
    train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(), inplace=True)
    test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(), inplace=True)
    train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0], inplace=True)
    test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0], inplace=True)

    # Consistent categories
    train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 
        'low fat': 'Low Fat', 
        'reg': 'Regular'
    })
    test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 
        'low fat': 'Low Fat', 
        'reg': 'Regular'
    })

    # Label Encoding
    le = LabelEncoder()
    for column in ['Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Fat_Content', 'Item_Type']:
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])

    # Feature Engineering
    train_data['Outlet_Age'] = 2024 - train_data['Outlet_Establishment_Year']
    test_data['Outlet_Age'] = 2024 - test_data['Outlet_Establishment_Year']

    # Drop unnecessary columns
    train_data = train_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)
    test_data = test_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1)
    
    return train_data, test_data
