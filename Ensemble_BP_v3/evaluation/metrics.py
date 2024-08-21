import numpy as np

def calculate_all_metrics(prediction, ground_truth):
    """
    Calculate and print MAE, RMSE, and Pearson Correlation for the given predictions and ground truth.
    Returns both the metrics and their standard errors.
    """
    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)
    num_test_samples = len(prediction)
    
    # MAE 계산
    mae = np.mean(np.abs(prediction - ground_truth))
    standard_error_mae = np.std(np.abs(prediction - ground_truth)) / np.sqrt(num_test_samples)
    print("MAE: {0:.4f} +/- {1:.4f}".format(mae, standard_error_mae))
    
    # RMSE 계산
    rmse = np.sqrt(np.mean(np.square(prediction - ground_truth)))
    standard_error_rmse = np.std(np.square(prediction - ground_truth)) / np.sqrt(num_test_samples)
    print("RMSE: {0:.4f} +/- {1:.4f}".format(rmse, standard_error_rmse))
    
    # Pearson Correlation 계산
    # pearson_corr = np.corrcoef(prediction, ground_truth)[0, 1]
    # standard_error_pearson = np.sqrt((1 - pearson_corr**2) / (num_test_samples - 2))
    # print("Pearson Correlation: {0:.4f} +/- {1:.4f}".format(pearson_corr, standard_error_pearson))
    
    return (mae, standard_error_mae), (rmse, standard_error_rmse) #, (pearson_corr, standard_error_pearson)

