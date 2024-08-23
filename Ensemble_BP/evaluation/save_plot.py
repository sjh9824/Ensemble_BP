import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def loss_plot(train_loss_list, valid_loss_list, plot_path):
    try:
        if len(train_loss_list) != len(valid_loss_list):
            print("The lengths of train loss and valid loss do not match.")

        else:
            min_loss = min(min(train_loss_list), min(valid_loss_list))
            max_loss = max(max(train_loss_list), max(valid_loss_list))
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_list, label='Train Loss')
            plt.plot(valid_loss_list, label='Validation Loss')
            plt.ylim(min_loss - 0.1*(max_loss-min_loss), max_loss + 0.1*(max_loss-min_loss))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train and Validation Loss over Epochs')
            plt.legend()
            plt.grid(True)

            # Save the plot
            save_path = os.path.join(plot_path, 'train_valid_loss_plot.png')
            plt.savefig(save_path)
            plt.close()
    except:
        print("\nError to save train loss graph\n")

def scatter_plots(predicted_sbp, predicted_dbp, ground_truth_sbp, ground_truth_dbp, plot_path, seed_list=[], plot_name = None):
        predicted_dbp = np.array(predicted_dbp)
        predicted_sbp = np.array(predicted_sbp)
        ground_truth_dbp = np.array(ground_truth_dbp)
        ground_truth_sbp = np.array(ground_truth_sbp)

        # MAE, RMSE, Pearson correlation coefficient 계산
        mae_sbp = np.mean(np.abs(predicted_sbp - ground_truth_sbp))
        rmse_sbp = np.sqrt(np.mean(np.square(predicted_sbp - ground_truth_sbp)))
        r_sbp, _ = pearsonr(ground_truth_sbp, predicted_sbp)

        mae_dbp = np.mean(np.abs(predicted_dbp - ground_truth_dbp))
        rmse_dbp = np.sqrt(np.mean(np.square(predicted_dbp - ground_truth_dbp)))
        r_dbp, _ = pearsonr(ground_truth_dbp, predicted_dbp)

        # Scatter plot 생성
        plt.figure(figsize=(14, 6))

        # SBP scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(ground_truth_sbp, predicted_sbp)
        plt.plot([ground_truth_sbp.min(), ground_truth_sbp.max()], [ground_truth_sbp.min(), ground_truth_sbp.max()], 'r-')
        plt.plot([ground_truth_sbp.min(), ground_truth_sbp.max()], [ground_truth_sbp.min()+10, ground_truth_sbp.max()+10], 'b--')
        plt.plot([ground_truth_sbp.min(), ground_truth_sbp.max()], [ground_truth_sbp.min()-10, ground_truth_sbp.max()-10], 'b--')
        plt.title('SBP: Predicted vs Ground Truth')
        plt.xlabel('Ground-truth')
        plt.ylabel('Prediction')
        plt.text(0.05, 0.95, f'MAE: {mae_sbp:.2f}\nRMSE: {rmse_sbp:.2f}\nr: {r_sbp:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

        # DBP scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(ground_truth_dbp, predicted_dbp)
        plt.plot([ground_truth_dbp.min(), ground_truth_dbp.max()], [ground_truth_dbp.min(), ground_truth_dbp.max()], 'r-')
        plt.plot([ground_truth_dbp.min(), ground_truth_dbp.max()], [ground_truth_dbp.min()+10, ground_truth_dbp.max()+10], 'b--')
        plt.plot([ground_truth_dbp.min(), ground_truth_dbp.max()], [ground_truth_dbp.min()-10, ground_truth_dbp.max()-10], 'b--')
        plt.title('DBP: Predicted vs Ground Truth')
        plt.xlabel('Ground-truth')
        plt.ylabel('Prediction')
        plt.text(0.05, 0.95, f'MAE: {mae_dbp:.2f}\nRMSE: {rmse_dbp:.2f}\nr: {r_dbp:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

        plt.tight_layout()
        print(plot_name)

        if plot_name is not None:
             file_name = plot_name + '_' + 'Scatter.png'
        else:
             file_name = 'Scatter.png'
        save_path = os.path.join(plot_path, file_name)

        plt.savefig(save_path)
        print(f'save plot at {save_path}')
        plt.close()

        