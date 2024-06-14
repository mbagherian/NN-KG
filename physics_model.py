import tensorflow as tf
import tensorflow_probability as tfp

def make_prior(kernel_size, bias_size, dtype=None):
    loc = tf.Variable(initial_value=tf.zeros((kernel_size + bias_size,), dtype=dtype), trainable=True)
    scale = tfp.util.TransformedVariable(
        initial_value=tf.random.normal((kernel_size + bias_size,), dtype=dtype) * 0.01,
        bijector=tfp.bijectors.Softplus(),
        dtype=dtype
    )
    def _create_prior(inputs):
        return tfp.distributions.Normal(loc=loc, scale=scale)
    return _create_prior

def make_posterior(kernel_size, bias_size, dtype=None):
    loc = tf.Variable(initial_value=tf.zeros((kernel_size + bias_size,), dtype=dtype), trainable=True)
    scale = tfp.util.TransformedVariable(
        initial_value=tf.random.normal((kernel_size + bias_size,), dtype=dtype) * 0.01,
        bijector=tfp.bijectors.Softplus(),
        dtype=dtype
    )
    def _create_posterior(inputs):
        return tfp.distributions.Normal(loc=loc, scale=scale)
    return _create_posterior

def build_physics_model():
    num_points = 1000
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(
            100, activation='relu', input_shape=(2,),
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        ),
        tfp.layers.DenseVariational(
            100, activation='tanh',
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        ),
        tfp.layers.DenseVariational(
            100, activation='sigmoid',
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        ),
        tfp.layers.DenseVariational(
            100, activation='relu',
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        ),
        tfp.layers.DenseVariational(
            100, activation='tanh',
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        ),
        tfp.layers.DenseVariational(
            2,
            make_prior_fn=make_prior, make_posterior_fn=make_posterior,
            kl_weight=1/num_points
        )  # Output [dphi_dt, dphi_dr]
    ])
    
    return model
