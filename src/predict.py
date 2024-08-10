import pandas as pd

def predict_and_save(model, test_data, output_path):
    predictions = model.predict(test_data)
    submission = pd.DataFrame({
        'Item_Outlet_Sales': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'.")
