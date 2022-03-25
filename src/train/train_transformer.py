from types import new_class
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import sys
sys.path.append(".")
from src.data.load_data import importData
from src.models.transformer import build_model

def trainTransformer(dataset):
    path_to_data = "./data/raw/"
    
    if dataset == 'mitbih':
        
        # dataset sepcific hyperparameters
        n_classes = 5
        head_size=256
        num_heads=2
        ff_dim=2
        num_transformer_blocks=2
        mlp_units=[256]
        mlp_dropout=0.4
        dropout=0.25
       
    elif dataset == 'ptbdb':

        # dataset sepcific hyperparameters
        n_classes = 2
        head_size=256
        num_heads=2
        ff_dim=2
        num_transformer_blocks=2
        mlp_units=[256]
        mlp_dropout=0.4
        dropout=0.25
        
    else:
        raise ValueError(dataset, 'is not a valid dataset. Choose mitbih or ptbdb')
    
    (X, Y, X_test, Y_test) = importData(path_to_data, dataset)

    # batch_size, epoch and iteration
    batch_size = 100
    n_iters = 20000
    num_epochs = int(n_iters/(len(X)/batch_size))

    input_shape = X.shape[1:]
    model = build_model(
        input_shape,
        n_classes,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        mlp_dropout,
        dropout,
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()
    
    reconstructed_model = keras.models.load_model("./models/keras_transformer_" + dataset + ".h5")
    if reconstructed_model:
        model = reconstructed_model
    
    #add callbacks for saving model, early stopping and dynamic learning rate
    checkpoint = ModelCheckpoint("./models/keras_transformer_" + dataset + ".h5", monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_sparse_categorical_accuracy", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="sparse_categorical_accuracy", mode="max", patience=3, verbose=2)

    callbacks = [early, checkpoint, redonplat]

    model.fit(
        X,
        Y,
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.evaluate(X_test, Y_test, verbose=1)

trainTransformer('ptbdb')