import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.initializers import VarianceScaling

class BaseLineModel:
    
    def __init__(self, modelparameter, metrics):
        self.params = modelparameter
        self.use_multi_gpu = self.params.use_multi_gpu
        self.num_classes = self.params.num_classes
        self.img_rows = self.params.scaled_row_dim
        self.img_cols = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
        self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        self.my_metrics = metrics
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                             kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(Conv2D(32, (3, 3), activation='relu',
                             kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu',
                            kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
    
    def compile_model(self, loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta()):
        self.model.compile(loss=loss,
              optimizer=optimizer,
              metrics=self.my_metrics)
    
    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator
    
    def learn(self, use_multiprocessing=True, workers= 8):
        return self.model.fit_generator(generator=self.training_generator,
                    validation_data=self.validation_generator,
                    epochs=self.params.n_epochs, 
                    use_multiprocessing=use_multiprocessing,
                    workers=workers)
    
    def score(self, use_multiprocessing=True, workers=8):
        return self.model.evaluate_generator(generator=self.validation_generator,
                                      use_multiprocessing=use_multiprocessing, 
                                      workers=workers)
    
    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y
    
    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)
    
    def load(self, modelinputpath):
        self.model = load_model(modelinputpath)