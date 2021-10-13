import numpy as np

def adam(w, dw, config=None):
    """
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    m = config['m']
    v = config['v']
    t = config['t'] + 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    learning_rate = config['learning_rate']

    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    mb = m / (1 - beta1 ** t)
    vb = v / (1 - beta2 ** t)
    next_w = w - learning_rate * mb / (np.sqrt(vb) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t
    print(config['t'])

    return next_w, config

def tes1t1():
    epochs = 100000
    x_0 = np.ones((4, 4))
    x_0 += 5
    cf = None
    x = x_0
    x_last = x
    for epoch in range(epochs):
        x_last = x
        x, cf = adam(x, 2 * x, cf)
        if np.linalg.norm(x-x_last)<= 1e-16:
            break


    print(x)
    pass

if __name__ == "__main__":
    tes1t1()