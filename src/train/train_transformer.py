from types import new_class
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data.load_data import importData
from src.models.transformer import build_model

def trainTransformer(hyperparameters, options):
    path_to_data = "./data/raw/"
    
        
    # dataset sepcific hyperparameters
    dataset = options['dataset_name']
    n_classes = hyperparameters['n_classes'][dataset]
    head_size = hyperparameters['head_size']
    num_heads = hyperparameters['num_heads']
    ff_dim = hyperparameters['ff_dim']
    num_transformer_blocks = hyperparameters['num_transformer_blocks']
    mlp_units = hyperparameters['mlp_units']
    mlp_dropout = hyperparameters['mlp_dropout']
    dropout = hyperparameters['dropout']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
       
    (X, Y, X_test, Y_test) = importData(path_to_data, dataset)


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
    
    try:
        model_name = 'Transformer_{}_{}_{}_{}_{}_{}_{}'.format(dataset, num_epochs, head_size, num_heads, num_transformer_blocks, ff_dim, mlp_units)
        reconstructed_model = keras.models.load_model(options['saved_model_path'] + model_name + ".h5")
        print('loaded wheights') 
        return reconstructed_model
    except IOError:
        print('training from scratch')

    #add callbacks for saving model, early stopping and dynamic learning rate
    checkpoint = ModelCheckpoint(options['saved_model_path'] + model_name + ".h5", monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
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
    
    return model

