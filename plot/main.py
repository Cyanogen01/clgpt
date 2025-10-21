import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
plt.rcParams['axes.linewidth'] = 1.5


def mean_squared_error(y_true, y_pred):
    return np.mean((np.ravel(y_true) - np.ravel(y_pred))**2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred)))

def MAX(y_true, y_pred):
    return np.max(np.abs(np.ravel(y_true) - np.ravel(y_pred)))


def main():
    #path = r'C:\Users\iCosMea Pro\SOC estimation\python project\1222_10deg\preds.csv'
    path = r'C:\Users\iCosMea Pro\SOC estimation\python project\1122_25deg\preds.csv'
    
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        y_test = np.array([float(i) for i in next(csv_reader)])
        y_pre = np.array([float(i) for i in next(csv_reader)])
        s2 = np.array([float(i) for i in next(csv_reader)])
        y_pred_LSMT = np.array([float(i) for i in next(csv_reader)])
        y_pred_CNN = np.array([float(i) for i in next(csv_reader)])
        y_pred_CNN_LSTM = np.array([float(i) for i in next(csv_reader)])
        y_pred_GRU = np.array([float(i) for i in next(csv_reader)])
        y_pred_CNN_GRU = np.array([float(i) for i in next(csv_reader)])

    write_flag = False
    zoom_out = not write_flag
    # zoom_out = False
    temp = 25
    time_sparse = 50

    CNN_RMSE = root_mean_squared_error(y_pred_CNN, y_test)
    CNN_MAE = MAE(y_pred_CNN, y_test)
    LSTM_RMSE = root_mean_squared_error(y_pred_LSMT, y_test)
    LSTM_MAE = MAE(y_pred_LSMT, y_test)
    GRU_RMSE = root_mean_squared_error(y_pred_GRU, y_test)
    GRU_MAE = MAE(y_pred_GRU, y_test)
    CNN_LSTM_RMSE = root_mean_squared_error(y_pred_CNN_LSTM, y_test)
    CNN_LSTM_MAE = MAE(y_pred_CNN_LSTM, y_test)
    CNN_GRU_RMSE = root_mean_squared_error(y_pred_CNN_GRU, y_test)
    CNN_GRU_MAE = MAE(y_pred_CNN_GRU, y_test)
    CL_GPT_RMSE = root_mean_squared_error(y_pre, y_test)
    CL_GPT_MAE = MAE(y_pre, y_test)


    CNN_LSTM_MAX = MAX(y_pred_CNN_LSTM, y_test)
    CNN_GRU_MAX = MAX(y_pred_CNN_GRU, y_test)
    CL_GPT_MAX = MAX(y_pre, y_test)

    # print('CNN_RMSE',CNN_RMSE)
    # print('CNN_MAE', CNN_MAE)
    # print('LSTM_RMSE', LSTM_RMSE)
    # print('LSTM_MAE', LSTM_MAE)
    # print('GRU_RMSE',GRU_RMSE)
    # print('GRU_MAE', GRU_MAE)
    print('CNN_LSTM_RMSE', CNN_LSTM_RMSE)
    print('CNN_LSTM_MAE', CNN_LSTM_MAE)
    print('CNN_LSTM_MAX', CNN_LSTM_MAX)
    print('\nCNN_GRU_RMSE', CNN_GRU_RMSE)
    print('CNN_GRU_MAE', CNN_GRU_MAE)
    print('CNN_GRU_MAX', CNN_GRU_MAX)
    print('\nCL_GPT_RMSE', CL_GPT_RMSE)
    print('CL_GPT_MAE', CL_GPT_MAE)
    print('CL_GPT_MAX', CL_GPT_MAX)


    plt.figure(figsize=(3.92, 2.36))
    plt.subplots_adjust(left=0.135, bottom=0.19, top=0.95)
    plt.plot(np.arange(len(y_test)) * time_sparse, y_test, c='k', lw=2, label='Ture')
    plt.plot(np.arange(len(y_test)) * time_sparse, y_pre.flatten(), color='r', lw=1, ls='--', marker='o', markevery=1, markersize=3,
              label='CL-GPT', alpha=0.9)
    plt.fill_between(np.arange(len(y_test)) * time_sparse,
                     y_pre.flatten() + 2 * np.sqrt(s2.flatten()),
                     y_pre.flatten() - 2 * np.sqrt(s2.flatten()), color='orange', alpha=0.7,
                     label='95% CI')
    
    plt.plot(np.arange(len(y_test)) * time_sparse, y_pred_CNN_LSTM.flatten(), color='c', lw=1, ls='--', marker='v', markevery=1, markersize=3,
              label='CNN-LSTM', alpha=0.9)

    plt.plot(np.arange(len(y_test)) * time_sparse, y_pred_CNN_GRU.flatten(), color='tab:brown', lw=1, ls='--', marker='X', markevery=1, markersize=3,
             label='CNN-GRU', alpha=0.9)
    
    plt.tick_params(labelsize=9)

    plt.grid()
    
    if not zoom_out:
        plt.legend(loc="best", fontsize=9)
        plt.xlabel('Times [s]')
        plt.ylabel('SOC')

    if write_flag:
        plt.savefig(str(temp) + 'deg_comparison.png')
        plt.savefig(str(temp) + 'deg_comparison.pdf')
        

    """error plot"""
    plt.figure(figsize=(3.92, 2.36))
    plt.subplots_adjust(left=0.17, bottom=0.19, top=0.95)
    plt.plot(np.arange(len(y_test)) * time_sparse, y_pre.flatten() - y_test, color='r', lw=1, ls='--', marker='o', markevery=1, markersize=3,
             label='CL-GPT', alpha=0.9)

    plt.plot(np.arange(len(y_test)) * time_sparse, y_pred_CNN_LSTM.flatten()  - y_test, color='c', lw=1, ls='--', marker='v', markevery=1, markersize=3,
              label='CNN-LSTM', alpha=0.9)
    

    plt.plot(np.arange(len(y_test)) * time_sparse, y_pred_CNN_GRU.flatten() - y_test, color='tab:brown', lw=1, ls='--', marker='X', markevery=1,
             markersize=3,
             label='CNN-GRU', alpha=0.9)
    plt.xlabel('Times [s]')
    plt.ylabel('Error')
    plt.tick_params(labelsize=9)

    plt.grid()
    plt.legend(loc="best", fontsize=9)

    if write_flag:
        plt.savefig(str(temp) + 'deg_error.png')
        plt.savefig(str(temp) + 'deg_error.pdf')

    plt.show()


if __name__ == '__main__':
    main()