import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-4
    config.value_lr = 1e-4
    config.critic_lr = 1e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 5
    config.dropout_rate = None

    config.tau = 0.0025  # For soft target updates.

    config.max_approx_method = "CEM"
    config.max_approx_hypers = (
        ("maxits", 10),
        ("N", 20),
        ("Ne", 10),
        ("sampleMethod", "Uniform")
    )

    return config
