import pandas as pd
import numpy as np

class LoadLabels:
    
    def __init__(self, path):
        self.labels_path = path
        self.label_names = {
            0:  "Nucleoplasm",  
            1:  "Nuclear membrane",   
            2:  "Nucleoli",   
            3:  "Nucleoli fibrillar center",   
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",   
            6:  "Endoplasmic reticulum",   
            7:  "Golgi apparatus",   
            8:  "Peroxisomes",   
            9:  "Endosomes",   
            10:  "Lysosomes",   
            11:  "Intermediate filaments",   
            12:  "Actin filaments",   
            13:  "Focal adhesion sites",   
            14:  "Microtubules",   
            15:  "Microtubule ends",   
            16:  "Cytokinetic bridge",   
            17:  "Mitotic spindle",   
            18:  "Microtubule organizing center",   
            19:  "Centrosome",   
            20:  "Lipid droplets",   
            21:  "Plasma membrane",   
            22:  "Cell junctions",   
            23:  "Mitochondria",   
            24:  "Aggresome",   
            25:  "Cytosol",
            26:  "Cytoplasmic bodies",
            27:  "Rods & rings"
        }
        self.reverse_train_labels = dict((v,k) for k,v in self.label_names.items())
        
    def fill_targets(self, row):
        row.Target = np.array(row.Target.split(" ")).astype(np.int)
        for num in row.Target:
            name = self.label_names[int(num)]
            row.loc[name] = 1
        return row
    
    def load_labels(self):
        train_labels = pd.read_csv(self.labels_path)
        for key in self.label_names.keys():
            train_labels[self.label_names[key]] = 0
        train_labels = train_labels.apply(self.fill_targets, axis=1)
        return train_labels, self.label_names, self.reverse_train_labels