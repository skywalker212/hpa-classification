from generators import DataGenerator, PredictGenerator
import pandas as pd
import numpy as np

def train_model(model, partitions, train_labels, train_path, parameter, preprocessor):
    target_names = train_labels.drop(["Target", "Id"], axis=1).columns
    
    predictions = []
    histories = []
    
    for i, partition in enumerate(partitions):
        print("training in partition ",i+1)
        
        training_generator = DataGenerator(partition['train'], train_labels, parameter, preprocessor)
        validation_generator = DataGenerator(partition['validation'], train_labels, parameter, preprocessor)
        predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)
        
        model.set_generators(training_generator, validation_generator)
        histories.append(model.learn())
        
        proba_predictions = model.predict(predict_generator)
        proba_predictions = pd.DataFrame(index = partition['validation'], data=proba_predictions, columns=target_names)
        
        predictions.append(proba_predictions)
        
    return predictions, histories