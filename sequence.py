
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

class SeqAE:

    def __init__(self, encoding_dim=100, embedding_dim=300, optimizer='nadam', loss='binary_crossentropy',
                 metrics=['binary_crossentropy'], checkpoint=True, cp_filename='checkpoint/chkp_giraffe.best.hdf5'):
        self.model = None
        self.encoding_dim = encoding_dim
        self.embedding_dim = embedding_dim
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.cp_filename = cp_filename

    def build(self):
        pass

    def _compile(self, optimizer='nadam', loss='mae', metrics=['mae']):
        print('Compiling...')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _summary(self):
        self.model.summary()

    def save_model(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s.json" % filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s.h5" % filename)
        print("Saved model to disk")

    def load_model(self, filename):
        json_file = open('%s.json' % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("%s.h5" % filename)
        print("Loaded model from disk")

    def fit(self, X_input_dic, y, epochs=10, batch_size=16, shuffle=True, stopped=False):
        callbacks_list = []
        if self.checkpoint:
            checkpoint = ModelCheckpoint(self.cp_filename, monitor='val_loss', verbose=1, save_best_only=True,
                                         mode='auto')
            callbacks_list.append(checkpoint)

        if stopped:
            self.model.load_weights(self.cp_filename)

        self.model.fit(X_input_dic, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)

    def get_model(self):
        return self.model
