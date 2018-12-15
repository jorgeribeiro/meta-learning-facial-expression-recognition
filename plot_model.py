from keras.utils import plot_model

def plot(model, filename):
	plot_model(
		model,
		to_file=filename + '.png',
		show_shapes=True
	)