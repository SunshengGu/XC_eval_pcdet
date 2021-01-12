from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf


def auroc(preds, labels):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def aupr(preds, labels):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary    t = tpr[idx]
    f = fpr[idx]

    return 0.5 * (1 - t) + 0.5 * f labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    _detection_error = lambda idx: 0.5 * (1 - tpr[idx]) + 0.5 * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))


def plot_roc(preds, labels, title="Receiver operating characteristic", save_path=None, thresh=0.1, measure='XQ',
             cls_name=""):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    save_path: where to save the plot
    thresh: threshold used to generate the XQ values for the boxes
    measure: the measure being used to identify TP/FP, can be XQ, class score, or others
    cls_name: the specific class of object being analyzed (e.g., car, pedestrian, cyclist)
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.4f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path is not None:
        fig_name = ""
        if cls_name != "":
            fig_name += "{}_".format(cls_name)
        fig_name += "ROC_using_{}_".format(measure)
        if measure != 'cls_score':
            fig_name += "thresh_{}".format(thresh)
        fig_name += ".png"
        plt.savefig("{}/{}".format(save_path, fig_name))
        plt.close()
    else:
        plt.show()


def plot_multi_roc(cls_name_list, preds, labels, preds_cls, labels_cls,
                   title="Receiver operating characteristic", save_path=None, thresh=0.1, measure='XQ', cls_name=""):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    save_path: where to save the plot
    thresh: threshold used to generate the XQ values for the boxes
    measure: the measure being used to identify TP/FP, can be XQ, class score, or others
    cls_name: the specific class of object being analyzed (e.g., car, pedestrian, cyclist)
    """
    labels_0, labels_1, labels_2 = labels_cls
    preds_0, preds_1, preds_2 = preds_cls

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)
    fpr_0, tpr_0, _ = roc_curve(labels_0, preds_0)
    fpr_1, tpr_1, _ = roc_curve(labels_1, preds_1)
    fpr_2, tpr_2, _ = roc_curve(labels_2, preds_2)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)
    tpr95_0 = fpr_at_95_tpr(preds_0, labels_0)
    tpr95_1 = fpr_at_95_tpr(preds_1, labels_1)
    tpr95_2 = fpr_at_95_tpr(preds_2, labels_2)

    # Compute AUROC
    roc_auc = auroc(preds, labels)
    roc_auc_0 = auroc(preds_0, labels_0)
    roc_auc_1 = auroc(preds_1, labels_1)
    roc_auc_2 = auroc(preds_2, labels_2)

    # Draw the plot
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC all classes = %0.4f' % roc_auc)
    plt.plot(fpr_0, tpr_0, color='cyan',
             lw=lw, label='AUROC %s = %0.4f' % (cls_name_list[0], roc_auc_0))
    plt.plot(fpr_1, tpr_1, color='crimson',
             lw=lw, label='AUROC %s = %0.4f' % (cls_name_list[1], roc_auc_1))
    plt.plot(fpr_2, tpr_2, color='darkgreen',
             lw=lw, label='AUROC %s = %0.4f' % (cls_name_list[2], roc_auc_2))
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':')
    plt.plot([tpr95, tpr95], [0, 1], color='darkorange', lw=lw, linestyle=':',
             label='FPR (95%% TPR) all classes= %0.4f' % tpr95)
    plt.plot([tpr95_0, tpr95_0], [0, 1], color='cyan', lw=lw, linestyle=':',
             label='FPR (95%% TPR) %s= %0.4f' % (cls_name_list[0], tpr95_0))
    plt.plot([tpr95_1, tpr95_1], [0, 1], color='crimson', lw=lw, linestyle=':',
             label='FPR (95%% TPR) %s= %0.4f' % (cls_name_list[1], tpr95_1))
    plt.plot([tpr95_2, tpr95_2], [0, 1], color='darkgreen', lw=lw, linestyle=':',
             label='FPR (95%% TPR) %s= %0.4f' % (cls_name_list[2], tpr95_2))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path is not None:
        fig_name = ""
        if cls_name != "":
            fig_name += "{}_".format(cls_name)
        fig_name += "ROC_using_{}_".format(measure)
        if measure != 'cls_score':
            fig_name += "thresh_{}".format(thresh)
        fig_name += ".png"
        plt.savefig("{}/{}".format(save_path, fig_name))
        plt.close()
    else:
        plt.show()


def plot_pr(preds, labels, title="Precision recall curve", save_path=None, thresh=0.1, measure='XQ', cls_name="",
            flip=False):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    save_path: where to save the plot
    thresh: threshold used to generate the XQ values for the boxes
    measure: the measure being used to identify TP/FP, can be XQ, class score, or others
    cls_name: the specific class of object being analyzed (e.g., car, pedestrian, cyclist)
    flip: indicating if the TF labels are flipped
    """

    # Compute values for curve
    if flip:
        labels = [1-a for a in labels]
        preds = [-a for a in preds]
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.4f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path is not None:
        fig_name = ""
        if flip:
            fig_name += "flipped_"
        if cls_name != "":
            fig_name += "{}_".format(cls_name)
        fig_name += "precision_recall_using_{}_".format(measure)
        if measure != 'cls_score':
            fig_name += "thresh_{}".format(thresh)
        fig_name += ".png"
        plt.savefig("{}/{}".format(save_path, fig_name))
        plt.close()
    else:
        plt.show()


def get_val(arr, indices):
    """
    :param arr: the array where values are stored
    :param indices: the indices we are interested in
    :return: values corresponding to the indices
    """
    ret_list = []
    for index in indices:
        ret_list.append(arr[index])
    return ret_list


def plot_multi_pr(cls_name_list, preds, labels, preds_cls, labels_cls, title="Precision recall curve", save_path=None,
                  thresh=0.1, measure='XQ', cls_name="", flip=False):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    save_path: where to save the plot
    thresh: threshold used to generate the XQ values for the boxes
    measure: the measure being used to identify TP/FP, can be XQ, class score, or others
    cls_name: the specific class of object being analyzed (e.g., car, pedestrian, cyclist)
    flip: indicating if the TF labels are flipped
    """
    if flip:
        labels = [1-a for a in labels]
        preds = [-a for a in preds]
        for i in range(3):
            labels_cls[i] = [1-a for a in labels_cls[i]]
            preds_cls[i] = [-a for a in preds_cls[i]]
    labels_0, labels_1, labels_2 = labels_cls
    preds_0, preds_1, preds_2 = preds_cls

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)
    precision_0, recall_0, _ = precision_recall_curve(labels_0, preds_0)
    prc_auc_0 = auc(recall_0, precision_0)
    precision_1, recall_1, _ = precision_recall_curve(labels_1, preds_1)
    prc_auc_1 = auc(recall_1, precision_1)
    precision_2, recall_2, _ = precision_recall_curve(labels_2, preds_2)
    prc_auc_2 = auc(recall_2, precision_2)

    # find the label for the instance with top score

    # top_score = np.argwhere(preds == np.amax(preds)).flatten().tolist()
    # top_score_0 = np.argwhere(preds_0 == np.amax(preds_0)).flatten().tolist()
    # top_score_1 = np.argwhere(preds_1 == np.amax(preds_1)).flatten().tolist()
    # top_score_2 = np.argwhere(preds_2 == np.amax(preds_2)).flatten().tolist()
    # top_label = get_val(labels, top_score)
    # top_label_0 = get_val(labels_0, top_score_0)
    # top_label_1 = get_val(labels_1, top_score_1)
    # top_label_2 = get_val(labels_2, top_score_2)
    # top_label_str = "top_score_label_for_all_{}_for_classes_{}_{}_{}".format(
    #     top_label, top_label_0, top_label_1, top_label_2
    # )

    plt.figure()
    lw = 1
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.4f)' % prc_auc)
    plt.plot(recall_0, precision_0, color='cyan',
             lw=lw, label='PR curve %s (area = %0.4f)' % (cls_name_list[0], prc_auc_0))
    plt.plot(recall_1, precision_1, color='crimson',
             lw=lw, label='PR curve %s (area = %0.4f)' % (cls_name_list[1], prc_auc_1))
    plt.plot(recall_2, precision_2, color='darkgreen',
             lw=lw, label='PR curve %s (area = %0.4f)' % (cls_name_list[2], prc_auc_2))
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path is not None:
        fig_name = ""
        if flip:
            fig_name += "flipped_"
        if cls_name != "":
            fig_name += "{}_".format(cls_name)
        fig_name += "precision_recall_using_{}_".format(measure)
        if measure != 'cls_score':
            fig_name += "thresh_{}_".format(thresh)
        # fig_name += top_label_str
        fig_name += ".png"
        plt.savefig("{}/{}".format(save_path, fig_name))
        plt.close()
    else:
        plt.show()


def get_summary_statistics(predictions, labels, thresh, measure, cls):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.

    These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.

        preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """

    return {
        'XQ_thresh': thresh,
        'measure': measure,
        'class': cls,
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels) * 100,
        'detection_error': detection_error(predictions, labels) * 100,
        'auroc': auroc(predictions, labels) * 100,
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels]) * 100,
        'aupr_in': aupr(predictions, labels) * 100
    }


def get_summary_statistics_wsum(predictions, labels, thresh, measure, cls, w_xq, w_cls_score):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.

    These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.

        preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.True
    """

    return {
        'w_xq': w_xq,
        'w_cls_score': w_cls_score,
        'XQ_thresh': thresh,
        'measure': measure,
        'class': cls,
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels) * 100,
        'detection_error': detection_error(predictions, labels) * 100,
        'auroc': auroc(predictions, labels) * 100,
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels]) * 100,
        'aupr_in': aupr(predictions, labels) * 100
    }