from hyperopt import hp

space_simplenet = {
	'activation': hp.choice('activation', ['relu', 'elu', 'tanh']),
	'batch_size': hp.quniform('batch_size', 10, 30, 5),
	'conv_filter_size_mult': hp.uniform('conv_filter_size_mult', 0.7, 1.4),
	'conv_kernel_initializer': hp.choice('conv_kernel_initializer', ['glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform']),
    'conv_kernel_size': hp.quniform('conv_kernel_size', 2, 4, 1),
	'conv_padding': hp.choice('conv_padding', ['same']),
	'data_aug': hp.choice('data_aug', [True]), # sempre usar data augmentation	
	'dropout': hp.uniform('dropout', 0.0, 0.35),
	'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'adadelta', 'adamax']),
	'pool_padding': hp.choice('pool_padding', ['same']),
    'pool_size': hp.quniform('pool_size', 2, 4, 1),
	'pool_strides': hp.quniform('pool_strides', 2, 4, 1),
	'use_BN': hp.choice('use_BN', [True]) # sempre usar batch normalization	
}

space_vgg_16 = {
	'activation': hp.choice('activation', ['relu', 'elu', 'tanh']),
	'batch_size': hp.quniform('batch_size', 20, 100, 10),
	'conv_filter_size_mult': hp.uniform('conv_filter_size_mult', 0.7, 1.4),
	'conv_kernel_initializer': hp.choice('conv_kernel_initializer', ['glorot_normal', 'glorot_uniform', 'random_normal', 'random_uniform']),
	'conv_kernel_size': hp.quniform('conv_kernel_size', 2, 4, 1),
	'conv_padding': hp.choice('conv_padding', ['same']),
    'data_aug': hp.choice('data_aug', [True, False]),
    'dropout': hp.uniform('dropout', 0.0, 0.5),
    'optimizer': hp.choice('optimizer', ['adam', 'adadelta', 'sgd']),
    'pool_padding': hp.choice('pool_padding', ['same']),
    'pool_size': hp.quniform('pool_size', 2, 4, 1),
    'pool_strides': hp.quniform('pool_strides', 2, 4, 1)
}