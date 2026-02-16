from kfp.v2.dsl import (
    component, 
    Input, 
    Output, 
    Dataset, 
    Metrics, 
    Model
)
from typing import NamedTuple
from typing import List

@component(
    base_image="python:3.11", 
    packages_to_install=[
      "numpy",
      "pandas",
      "keras",
      "tensorflow",
      "scikit-learn"
    ]
)
def train_vae(
    dataset: Input[Dataset],
    seq_length,
    n_features,    # PRICE, PDCC_down, OSV_down, PDCC_up, OSV_up
    latent_dim: int = 3,
    initial_kl_weight: int = 1e-4,
    # Thresholds
    # thresholds = [0.001, 0.005, 0.01, 0.015],
    threshold, # Threshold to train on
    epochs: int = 200,
    z: Output[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    # Input batch: (batch_size, window_size, n_features)
    # Encoded batch: (batch_size, latent_dim)
    # Decoded batch: (batch_size, window_size, n_features)
    import datetime
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers, ops
    from sklearn.model_selection import train_test_split

    """
    ## Create a sampling layer
    """
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seed_generator = keras.random.SeedGenerator(1998)

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = ops.shape(z_mean)[0]
            dim   = ops.shape(z_mean)[1]
            eps   = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var) * eps

    """
    ## Build the encoder
    """
    # OLD: Input(shape=(1, seq_length))
    # NEW:
    encoder_inputs = keras.Input(shape=(seq_length, n_features))

    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(encoder_inputs)
    x = layers.LSTM(64,  return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=f"encoder")
    encoder.summary()
    # keras.utils.plot_model(encoder, f'plots/vae_encoder_{threshold}.png', show_shapes=True)


    """
    ## Build the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    # OLD: RepeatVector(1)
    # NEW:
    x = layers.RepeatVector(seq_length)(latent_inputs)

    x = layers.LSTM(32,  return_sequences=True)(x)
    x = layers.LSTM(64,  return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)

    # OLD: Dense(seq_length)
    # NEW: produce n_features at each timestep
    decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    # keras.utils.plot_model(decoder, f'plots/vae_decoder_{threshold}.png', show_shapes=True)


    """
    ## Define the VAE as a `Model` with a custom `train_step`
    """
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, kl_weight=initial_kl_weight, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.kl_weight = kl_weight

            # Trackers for loss componenets
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker]

        def train_step(self, data):
            # now data shape = (batch_size, seq_length, n_features)
            # Forward pass
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data, training=True)
                reconstruction = self.decoder(z, training=True)

                # Calculate reconstruction loss (MSE)
                # MSE over all timesteps & features
                # recon_loss = tf.reduce_mean(
                #     tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction), axis=[1,2])
                # )
                error = tf.square(data - reconstruction)
                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(error, axis=[1,2])
                )


                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = self.kl_weight * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                loss = recon_loss + kl_loss

            # Backprop and update
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # Update KL weight dynamically
            self.kl_weight = min(1.0, self.kl_weight * 1.10) # Gradually increase KL contribution

            # Update metrics
            self.total_loss_tracker.update_state(loss)
            self.reconstruction_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def test_step(self, data):
            """
            Similar to train_step, but no gradient updates.
            Used during validation/test phases (model.evaluate() or val_data in model.fit()).
            """
            z_mean, z_log_var, z = self.encoder(data, training=False)
            reconstruction = self.decoder(z, training=False)

            # Calculate reconstruction loss (MSE)
            # recon_loss = tf.reduce_mean(
            #     tf.reduce_sum(keras.losses.mean_squared_error(data, reconstruction), axis=[1,2])
            # )

            error = tf.square(data - reconstruction)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(error, axis=[1,2])
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.kl_weight * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            loss = recon_loss + kl_loss

            # Update metrics (no backprop here)
            self.total_loss_tracker.update_state(loss)
            self.reconstruction_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            # Return a dict mapping metric names to current value
            return {
                    "loss": loss, 
                    "reconstruction_loss": recon_loss, 
                    "kl_loss": kl_loss
            }

        def call(self, inputs, training=False):
            """
            This method defines the forward pass for inference.
            Typically, you'd want to return the reconstruction (or z if you prefer).
            """
            z_mean, z_log_var, z = self.encoder(inputs, training=training)
            reconstruction = self.decoder(z, training=training)
            return reconstruction


    # """
    # ## Split data
    # """
    # # Split into train/val as needed:
    # from sklearn.model_selection import train_test_split
    # X_train, X_val = train_test_split(vae_inputs[threshold], test_size=0.2, shuffle=False)
    # print("X_train:", X_train.shape, X_train.dtype)
    # print("X_val:  ",   X_val.shape,   X_val.dtype)

    """
    ## Callbacks
    """
    # # Tensorboard
    # log_dir = "logs/fit/vae_thr{:.3f}_".format(threshold) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_cb = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir,
    #     histogram_freq=1,
    #     write_graph=True,
    #     update_freq="epoch",
    # )

    # Model Checkpoint
    checkpoint_filepath = f"./vae_thr{threshold:.3f}_best.weights.h5"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    # Early stop
    earlystopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=2
    )

    # Fixed Learning rate
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3)


    """
    ## Build and compile
    """
    vae = VAE(encoder, decoder)
    vae.build(input_shape=(None, seq_length, n_features))
    vae.compile(optimizer=optimizer)

    """
    ## Train VAE
    """
    # Setup 

    history = vae.fit(
        X_train,                         # shape: (n_train, 50, 7)
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint_cb, earlystopping_cb]
    )
    
    """
    ## Evaluate VAE
    """    
    eval_results = vae.evaluate(X_train, return_dict=True)

    # eval_results = vae.evaluate(X_val, return_dict=True)

    # Save model
    vae.save(model.path)

    # Log each metric so that KFP UI picks it up
    for name, value in eval_results.items():
        # name: e.g. "loss", "reconstruction_loss", "kl_loss"
        # value: a Python float
        metrics.log_metric(name=name, number_value=float(value))
