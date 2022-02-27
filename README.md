# LogoGAN
This repository serves the purpose of creating logos using Generative Adversial Networks. 

This project uses GANs models to produce new logo images. It works by trying to mimic real-world images collected from the Wikipedia pages. More details about GANs are provided in the overview.


## GANs overview
Generative Adversarial Networks (GANs) belongs to the generative models. That means they are able to generate artificial content base on the arbitrary input.

Generally, GANs most of the time refers to the training method, rather on the generative model. Reason for this is that GANs don't train a single network, but instead two networks simultaneously.

The first network is usually called Generator, while the second Discriminator. Purpose of the Generator model is to images that look real. During training, the Generator progressively becomes better at creating images that look real. Purpose of the Discriminator model is to learn to tell real images apart from fakes. During training, the Discriminator progressively becomes better at telling fake images from real ones. The process reaches equilibrium when the Discriminator can no longer distinguish real images from fakes.


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


## Contact
Please, feel free to reach out on [LinkedIn](https://www.linkedin.com/in/tim-cvetko-32842a1a6/) or [Gmail](tim@metawaveai.com).

## License

Licensed under the MIT [LICENSE](LICENSE)

# Acknowledgments

Coursera Specialization by Deep Learning : https://www.coursera.org/specializations/generative-adversarial-networks-gans
