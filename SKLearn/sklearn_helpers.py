from sklearn.metrics import classification_report

def get_classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict = True)
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}