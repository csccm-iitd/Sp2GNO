import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np



def extract_mse_data(file_path):
    train_mse = []
    test_mse = []
    with open(file_path, 'r') as file:
        for line in file:
            if "train_mse:" in line and "test_mse:" in line:
                # Extract the train and test MSE values
                parts = line.split(" ")
                train_mse_value = float(parts[parts.index("train_mse:") + 1])
                test_mse_value = float(parts[parts.index("test_mse:") + 1])
                train_mse.append(train_mse_value)
                test_mse.append(test_mse_value)
    return train_mse, test_mse




def export_excel(log_file_path, excel_file_path):
    train_mse, test_mse = extract_mse_data(log_file_path)
    max_length = len(train_mse)
    data = {
    "Epoch": list(range(max_length)),
    "Train_MSE": train_mse,
    "Test_MSE": test_mse,
    }

    df = pd.DataFrame(data)
    df.to_excel(excel_file_path, index=False)
    print("excell data saved at",excel_file_path)


# if __name__ == "__main__":
#     a =0
    
