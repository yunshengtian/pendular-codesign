config = {
    
    'ilqr': {
        'max_iter': 100,
        'regu_init': 1,
        'u_init_sigma': 0.0001,
    },

    'mppi': {
        'K': 1000,
        'T': 10,
        'lambda': 1,
        'noise_mu': 0.0,
        'noise_sigma': 10.0,
        'u_init_sigma': 1.0,
        'num_cores': 1,
    },
}
