import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import os


parser = argparse.ArgumentParser(description='This script compare files with predicted data and actual data for '
                                             'circuits configured in a home. This script expects predicted file name,'
                                             'actual data file name and base directory as input arguments.')
parser.add_argument("--input_file", help="File with actual data.", type=str)
parser.add_argument("--test_file", help="File with predicted data.", type=str)
parser.add_argument("--base_dir", help="Path of base directory where input files are stored and output file "
                                       "will be generated.", type=str)
args = parser.parse_args()

truefile = args.input_file
testfile = args.test_file
project_dir = args.base_dir

# MSE Check on the circuit
def calc_mse(actual, pred):
    return mean_squared_error(actual, pred)


# Root Mean Square Error check
def calc_rmse(mse):
    return sqrt(mse)


# Mean Absolute Error:
def calc_mae(actual, pred):
    return mean_absolute_error(actual, pred)


def calc_r2(actual, pred):
    return r2_score(actual, pred)


def validate_state(read_true_file , read_pred_file , ckt):

    read_true_file['state'] = np.where(read_true_file[ckt] >= 0.015, 'ON', 'OFF')

    # Identify the state of circuits for predicted data
    read_pred_file['state'] = np.where(read_pred_file[ckt] >= 0.015, 'ON', 'OFF')

    return read_true_file['state'],read_pred_file['state']


def calc_average_for_summary(input_dic):
    # Calculate average mse for all circuits:
    if len(input_dic) > 0:
        input_dic_sum = 0
        for key, value in input_dic.items():
            input_dic_sum = input_dic_sum + value
        input_dic_avg = input_dic_sum / len(mse_dict)
    return input_dic_avg

if('sample' in truefile):
    read_true_file = pd.read_csv(project_dir + '/sample_files/' + truefile)
    read_pred_file = pd.read_csv(project_dir + '/sample_files/' + testfile)
    dataid = (truefile.split('_')[2]).split('.')[0]
    out_dir = project_dir + '/sample_files/'
else:
    read_true_file = pd.read_csv(project_dir + truefile)
    read_pred_file = pd.read_csv(project_dir + testfile)
    dataid = (truefile.split('_')[1]).split('.')[0]
    out_dir = project_dir

# Read metadata file
metadata_file = pd.read_csv(project_dir + "metadata_file.csv")

# Create output file
if os.path.exists(out_dir + "summary_output_" + str(dataid) + ".txt"):
    os.remove(out_dir + "summary_output_" + str(dataid) + ".txt")

out_file = open(out_dir + "summary_output_" + str(dataid) + ".txt", "a")


print("Processing File for dataid " + dataid + "...")

# list of circuits configured in the given home.
configured_ckts = (metadata_file[dataid].values.tolist())

# create new dataframe for absolute errors
absolute_errors = pd.DataFrame()
absolute_errors['localminute'] = read_pred_file['localminute']
predicted_ckts = []
wrongly_predicted_ckts = []
right_predicted_ckts = []
mse_dict = {}
rmse_dict = {}
mae_dict = {}
r2_dict = {}
accuracy_scores = {}
precision = {}
recall = {}
f1score = {}
cm = {}

for predicted_ckt in read_pred_file.columns:
    predicted_ckts.append(predicted_ckt)

for ckt in predicted_ckts:
    if ckt not in configured_ckts and ckt != 'localminute' and ckt != 'dataid' and ckt != 'grid':
        wrongly_predicted_ckts.append(ckt)

    if ckt in configured_ckts and ckt != 'localminute' and ckt != 'dataid' and ckt != 'grid':
        right_predicted_ckts.append(ckt)
        print("Comparing " + ckt + "...")

        # Mean Squared Error
        mse_dict[ckt] = round(calc_mse(read_true_file[ckt], read_pred_file[ckt]), 4)

        # Root Mean squared Error
        rmse_dict[ckt] = round(calc_rmse(mse_dict[ckt]), 4)

        # Mean Absolute Error
        mae_dict[ckt] = round(calc_mae(read_true_file[ckt], read_pred_file[ckt]), 4)

        # R2 Score
        r2_dict[ckt] = round(calc_r2(read_true_file[ckt], read_pred_file[ckt]), 4)

        # Absolute Error for all values
        absolute_errors[ckt] = round(abs(read_true_file[ckt] - read_pred_file[ckt]), 4)

        # Identify the state of circuits for actual data
        true_state, pred_state = validate_state(read_true_file, read_pred_file, ckt)

        # Calculate Accuracy Score
        accuracy_scores[ckt] = round(accuracy_score(true_state, pred_state), 5)

        # Calculate precision score:
        precision[ckt] = round(precision_score(true_state, pred_state, pos_label='ON'), 5)

        # Calculate Recall score:
        recall[ckt] = round(recall_score(true_state, pred_state, pos_label='ON'), 5)

        # Cacluate F1 score:
        f1score[ckt] = round(f1_score(true_state, pred_state, pos_label='ON'), 5)

        # Compute confusion matrix
        cm[ckt] = confusion_matrix(true_state, pred_state)

# Generate summary
if len(wrongly_predicted_ckts) > 0:
    print("**You have wrongly predicted the circuits mentioned below. These are not configured in home " + dataid +
          " and thus will not be validated...", file=out_file)
    for ckt in wrongly_predicted_ckts:
        print("->" + ckt, file=out_file)
    print("\n", file=out_file)

mse_avg = calc_average_for_summary(mse_dict)
mae_avg = calc_average_for_summary(mae_dict)
rmse_avg = calc_average_for_summary(rmse_dict)
r2_avg = calc_average_for_summary(r2_dict)

print("**Below are the averages of each metric over all circuits predicted...", file=out_file)
print("Average of mean square error : ", str(mse_avg), file=out_file)
print("Average of mean absolute error : ", str(mae_avg), file=out_file)
print("Average of root mean square error : ", str(rmse_avg), file=out_file)
print("Average of R2 score : ", str(r2_avg), file=out_file)
print("\n", file=out_file)


# Appliances that reported accuracy more than 50%
if len(accuracy_scores) > 0:
    print("**Below is the list of appliances that reported accuracy more than 50%...", file=out_file)
    for key, value in accuracy_scores.items():
        if value > 0.5:
            print(key + ' -> ' + str(value), file=out_file)
        else:
            continue
    print("\n", file=out_file)


# Below are the classification metrics for each of the predicted circuit:
print("**Below are some of the metrics for each of the predicted circuit that will help you evaluate the "
      "performance of your algorithm. Please note that if circuit power is less than 15W then appliance has been "
      "considered as OFF for that interval.", file=out_file)
print("\n", file=out_file)
for ckt in right_predicted_ckts:
    print(ckt.upper() + ':', file=out_file)
    print("Mean Absolute Error:" + str(mae_dict[ckt]), file=out_file)
    print("Mean Square Error:" + str(mse_dict[ckt]), file=out_file)
    print("Root Mean squared Error:" + str(rmse_dict[ckt]), file=out_file)
    print("R2 Score:" + str(r2_dict[ckt]), file=out_file)
    print("Accuracy Score:" + str(accuracy_scores[ckt]), file=out_file)
    print("Precision:" + str(precision[ckt]), file=out_file)
    print("Recall Score:" + str(recall[ckt]), file=out_file)
    print("F1 Score:" + str(f1score[ckt]), file=out_file)
    print("\n", file=out_file)

# Export absolute data
export_absolute_error_data = absolute_errors.to_csv(out_dir + 'absolute_errors_' + dataid + '.csv',
                                                    index=None, header=True)
