from sklearn.metrics import f1_score


def macro_f1(trues, preds):
    return f1_score(trues, preds, average='macro')
