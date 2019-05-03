import numpy as np
import pandas as pd

class PredictGenerator:
    
    def __init__(self, predict_Ids, imagepreprocessor, predict_path):
        self.preprocessor = imagepreprocessor
        self.preprocessor.basepath = predict_path
        self.identifiers = predict_Ids
    
    def predict(self, model):
        y = np.empty(shape=(len(self.identifiers), self.preprocessor.parameter.num_classes))
        for n in range(len(self.identifiers)):
            image = self.preprocessor.preprocess(self.identifiers[n])
            image = image.reshape((1, *image.shape))
            y[n] = model.predict(image)
        print(len(y))
        return y
    
def transform_to_target(row):
    target_list = []
    for col in baseline_labels.drop(["Target"], axis=1).columns:
        if row[col] == 1:
            target_list.append(str(reverse_train_labels[col]))
    if len(target_list) == 0:
        return str(0)
    return " ".join(target_list)   

def gen_test(train_labels, test_parameter, test_preprocessor, model, title, label_names, sub_path="./dataset/sample_submission.csv",test_path="./dataset/test/"):
    submission = pd.read_csv(sub_path)
    print('read submission file')
    test_names = submission.Id.values
    test_labels = pd.DataFrame(data=test_names, columns=["Id"])
    for col in train_labels.columns.values:
        if col != "Id":
            test_labels[col] = 0
    print('formatted dataframer for submission')
    submission_predict_generator = PredictGenerator(test_names, test_preprocessor, test_path)
    print('created predict generator, predicting test results')
    submission_proba_predictions = model.predict(submission_predict_generator)
    print('prediction of test labels done')
    baseline_labels = test_labels.copy()
    baseline_labels.loc[:, test_labels.drop(["Id", "Target"], axis=1).columns.values] = submission_proba_predictions
    baseline_labels.to_csv(title+"_baseline_submission_proba.csv")
    reverse_train_labels = dict((v,k) for k,v in label_names.items())
    for i in range(1,6):
        thres = i / 10
        for column in baseline_labels.drop(['Id','Target'],axis=1).columns.values:
            baseline_labels.loc[:,column] = np.where(baseline_labels.loc[:,column]>=thres,1,0)
        baseline_labels["Predicted"] = baseline_labels.apply(lambda l: transform_to_target(l), axis=1)
        submission = baseline_labels.loc[:, ["Id", "Predicted"]] 
        submission.to_csv("test_"+title+"_0"+str(i)+".csv", index=False)
        print('saving formatted submission file for '+title+' with thresh '+str(thres)+' done')