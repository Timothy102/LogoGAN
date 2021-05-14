# LogoGAN
This repository serves the purpose of creating logos using Generative Adversial Networks. 



```python

    def generator():
        start = time.time()

        model = keras.Sequential([
            layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=(GEN_NOISE_INPUT_SHAPE,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),

            layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=False,
                                   activation="tanh"),
        ])
```
