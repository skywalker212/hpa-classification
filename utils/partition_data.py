from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
import numpy as np

class PartitionData:
    
    def __init__(self, train_labels):
        self.train_labels = train_labels
        
    def RepeatedKFold(self, n_splits, n_repeats, random_state):
        partitions = []
        
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        for train_idx, test_idx in splitter.split(self.train_labels['Id'].index.values):
            partition = {}
            partition["train"] = self.train_labels.Id.values[train_idx]
            partition["validation"] = self.train_labels.Id.values[test_idx]
            partitions.append(partition)
            
        return partitions
    
    def RepeatedMultilabelStratifiedKFold(self, n_splits, n_repeats, random_state):
        partitions = []
        
        rmskf = RepeatedMultilabelStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        for train_index, test_index in rmskf.split(self.train_labels["Id"].index.values, self.train_labels.drop(columns=['Id','Target']).values):
            partition = {}
            partition["train"] = self.train_labels.Id.values[train_index]
            partition["validation"] = self.train_labels.Id.values[test_index]
            partitions.append(partition)
            
        return partitions
    
    def MultilabelStratifiedShuffleSplit(self, n_splits, test_size, random_state):
        partitions = []
        
        msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        
        for train_index, test_index in msss.split(self.train_labels["Id"].index.values, self.train_labels.drop(columns=['Id','Target']).values):
            partition = {}
            partition["train"] = self.train_labels.Id.values[train_index]
            partition["validation"] = self.train_labels.Id.values[test_index]
            partitions.append(partition)
            
        return partitions