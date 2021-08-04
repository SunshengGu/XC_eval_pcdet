import os
import csv
import numpy as np
import pandas as pd


def main():
    source_dir = "/media/sg/02f4ed99-ea7d-47a9-9aed-3606f0fd7fda/tadenoud/Documents/WISEOpenLidarPerceptron/tools/XAI_results/XC_Aug3_Sum_positive"
    file_name = "/interested_pred_scores_XC_sum_pos.csv"
    tp_file_name = "/tp_xq_thresh0.1.csv"
    fp_file_name = "/fp_xq_thresh0.1.csv"
    file_path = source_dir + file_name
    tp_path = source_dir + tp_file_name
    fp_path = source_dir + fp_file_name
    field_names = ['batch', 'pred_label', 'pred_score', 'xc', 'far_attr', 'pap']
    tp_csv = open(tp_path, 'w', newline='')
    tp_writer = csv.DictWriter(tp_csv, delimiter=',', fieldnames=field_names)
    tp_writer.writeheader()
    fp_csv = open(fp_path, 'w', newline='')
    fp_writer = csv.DictWriter(fp_csv, delimiter=',', fieldnames=field_names)
    fp_writer.writeheader()

    all_pred_data = pd.read_csv(file_path)
    for i in range(len(all_pred_data['tp/fp'])):
        data_dict = {"batch": all_pred_data['batch'][i], "pred_label": all_pred_data['pred_label'][i],
                     "pred_score": all_pred_data['pred_score'][i], "xc": all_pred_data['xc'][i],
                     "far_attr": all_pred_data['far_attr'][i], "pap": all_pred_data['pap'][i]}
        if all_pred_data['tp/fp'][i] == 'tp':
            tp_writer.writerow(data_dict)
        else:
            fp_writer.writerow(data_dict)

if __name__ == '__main__':
    main()