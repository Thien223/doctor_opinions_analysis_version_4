import numpy as np
import random
import torch
torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1542)
np.random.seed(1542)
from torch.nn import Dropout
from torch.nn import functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
######################
####### modules#######
######################
class LinearNorm(torch.nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
		super(LinearNorm, self).__init__()
		self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

		torch.nn.init.xavier_uniform_(
			self.linear_layer.weight,
			gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, x):
		return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
				 padding=None, dilation=1, bias=True, w_init_gain='linear'):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)

		self.conv = torch.nn.Conv1d(in_channels, out_channels,
									kernel_size=kernel_size, stride=stride,
									padding=padding, dilation=dilation,
									bias=bias)

		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal



class Inception(torch.nn.Module):
	def __init__(self, in_channels=8, bottneck_out_channel=8, conv_out_channels=32):
		super(Inception, self).__init__()
		self.conv1d_10 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=3, stride=2)
		self.conv1d_20 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=9, stride=5)
		self.conv1d_40 = ConvNorm(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=27, stride=11)
		#### residual_conv and bottleneck convolution must match the inputs shape [batchsize, in_channel, with, height]
		self.bottleneck = ConvNorm(in_channels=in_channels, out_channels=bottneck_out_channel, kernel_size=1, stride=1)
		self.residual_conv = ConvNorm(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=1, stride=1)
		self.max_pooling = torch.nn.MaxPool1d(kernel_size=3, stride=1)
		self.batch_norm = torch.nn.BatchNorm1d(conv_out_channels)

	def forward(self, inputs):
		pool_out = self.max_pooling(inputs)
		residual_out = self.residual_conv(pool_out)

		bottleneck_output = self.bottleneck(inputs)

		conv_10_out = self.conv1d_10(bottleneck_output)

		conv_20_out = self.conv1d_20(bottleneck_output)

		conv_40_out = self.conv1d_40(bottleneck_output)

		conv_outs = torch.cat((conv_10_out,conv_20_out,conv_40_out,residual_out), dim=2)
		output = self.batch_norm(conv_outs)
		return output

class Encoder(torch.nn.Module):
	def __init__(self, hparams=None):
		super(Encoder, self).__init__()
		# self.hidden_size = hparams.lstm_hiddent_size
		self.linear = LinearNorm(in_dim=13120, out_dim=128)
		self.inception_1 = Inception(in_channels=hparams.input_sequence_length, bottneck_out_channel=hparams.bottneck_outchannel, conv_out_channels=hparams.inception_inchannel)
		# self.linear_batch_norm = torch.nn.BatchNorm1d(8)
		# self.relu = torch.nn.ReLU()
		self.inception_2 = Inception(in_channels=hparams.inception_inchannel, bottneck_out_channel=hparams.bottneck_outchannel, conv_out_channels=hparams.inception_inchannel)
		# self.linear2 = LinearNorm(in_dim=128, out_dim=256)
		# self.linear3 = LinearNorm(in_dim=256, out_dim=64)
		self.dropout_1 = Dropout(p=0.4,inplace=True)
		self.dropout_2 = Dropout(p=0.2, inplace=True)

	def forward(self, inputs, debug=True):
		# linear_out = self.dropout_1(F.relu(self.linear(inputs.float())))
		# linear_out = self.dropout_2(F.relu(self.linear2(linear_out)))
		# linear_out = self.linear3(linear_out)
		inputs = self.inception_1(inputs)

		inputs=self.dropout_1(inputs)
		inputs = self.inception_2(inputs)

		inputs=self.dropout_2(inputs)
		inputs = inputs.view(-1, inputs.size(1)*inputs.size(2))
		linear_out = self.linear(inputs)

		#print(f'encoder ougput.shape {linear_out.shape}')
		return linear_out

class Decoder(torch.nn.Module):
	def __init__(self, hparams=None):
		super(Decoder, self).__init__()
		self.hparams = hparams
		# self.linear_batch_norm = torch.nn.BatchNorm1d(self.hparams.linear_in_dim)
		# self.linear_relu = torch.nn.ReLU()
		self.linear = LinearNorm(in_dim=128, out_dim=13120)
		self.conv = ConvNorm(in_channels=1644, out_channels=512)
		self.upsample_1 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=64, kernel_size=3)
		self.upsample_2 = torch.nn.ConvTranspose1d(in_channels=64, out_channels=150, kernel_size=3)
		self.dropout = Dropout(p=0.4, inplace=True)

	def forward(self, inputs_, debug=False):

		#print(f'inputs_.shape {inputs_.shape}')
		epsion = torch.zeros(size=inputs_.size(), device=device).normal_(mean=0.0, std=0.01)  #### noise value to add to encoder output
		inputs = inputs_ + epsion
		out =self.dropout(F.relu(self.linear(inputs)))
		#print(f'out.shape {out.shape}')
		out=out.reshape(out.size(0), self.hparams.inception_inchannel, -1)
		out = self.upsample_1(out)

		out=self.upsample_2(out)
		out = out.transpose(1,2)
		decoder_output =self.conv(out)
		decoder_output=decoder_output.transpose(1,2)
		return decoder_output, inputs_


class Generator(torch.nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.modules_ = torch.nn.Sequential(*[
			LinearNorm(in_dim=480, out_dim=128),
			torch.nn.Dropout(0.3,inplace=True),
			torch.nn.LeakyReLU(inplace=True),
			LinearNorm(in_dim=128, out_dim=256),
			torch.nn.Dropout(0.3,inplace=True),
			torch.nn.LeakyReLU(inplace=True),
			LinearNorm(in_dim=256, out_dim=480)
		])
		self.output_ = torch.nn.Tanh()

	def forward(self, latent_features):
		outputs = self.modules_(latent_features)
		outputs= self.output_(outputs)
		return outputs



class Discriminator(torch.nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.modules_ = torch.nn.Sequential(*[
			LinearNorm(in_dim=480, out_dim=256),
			torch.nn.Dropout(0.3, inplace=True),
			torch.nn.LeakyReLU(inplace=True),
			LinearNorm(in_dim=256, out_dim=128),
			torch.nn.Dropout(0.3, inplace=True),
			torch.nn.LeakyReLU(inplace=True),
			LinearNorm(in_dim=128, out_dim=64)
		])
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, inputs):
		outputs = self.modules_(inputs)
		# outputs = self.sigmoid(outputs)
		return outputs

class Auxiliary_Classifier(torch.nn.Module):
	def __init__(self, hparams):
		super(Auxiliary_Classifier, self).__init__()
		self.hidden_size = hparams.lstm_hiddent_size
		self.linear1 = LinearNorm(in_dim=32, out_dim=16)
		self.linear2 = LinearNorm(in_dim=480, out_dim=7)
		# self.linear3 = LinearNorm(in_dim=13120, out_dim=64)
		self.inception_1 = Inception(in_channels=hparams.input_sequence_length, bottneck_out_channel=hparams.bottneck_outchannel, conv_out_channels=hparams.inception_inchannel)
		# self.linear_batch_norm = torch.nn.BatchNorm1d(8)
		self.relu = torch.nn.ReLU()
		self.inception_2 = Inception(in_channels=hparams.inception_inchannel, bottneck_out_channel=hparams.bottneck_outchannel, conv_out_channels=hparams.inception_inchannel)
		# self.linear2 = LinearNorm(in_dim=2048, out_dim=512)
		# self.linear3 = LinearNorm(in_dim=512, out_dim=64)
		self.dropout = Dropout(p=0.4, inplace=True)
		# self.dropout_2 = Dropout(p=0.2, inplace=True)

	def forward(self, inputs_):
		inputs = inputs_
		# inputs = self.inception_1(inputs)
		# inputs = self.inception_2(inputs)
		#
		# inputs = inputs.view(-1, inputs.size(1) * inputs.size(2))

		# linear_out = self.dropout(self.linear1(inputs))
		linear_out = self.linear2(inputs)
		return linear_out

#################################
###### classifying models #######
#################################
class Unilabel_Classifier(torch.nn.Module):
	def __init__(self, words_count, hparams):
		super(Unilabel_Classifier, self).__init__()
		inception_inchannel = hparams.inception_inchannel
		bottneck_outchannel = hparams.bottneck_outchannel
		input_sequence_length = hparams.input_sequence_length
		num_classes = hparams.output_sequence_length
		linear_in_dim = hparams.linear_in_dim

		self.inception = Inception(in_channels=inception_inchannel, bottneck_out_channel=bottneck_outchannel, conv_out_channels=inception_inchannel)
		self.linear = LinearNorm(in_dim=linear_in_dim, out_dim=num_classes)
		self.conv1d = ConvNorm(in_channels=input_sequence_length, out_channels=inception_inchannel, kernel_size=3, stride=1, w_init_gain='relu')
		self.softmax = torch.nn.Softmax(dim=1)
		# self.relu = torch.nn.ReLU()
		self.embedding = torch.nn.Embedding(embedding_dim=512, num_embeddings=words_count + 10)
		self.batch_norm = torch.nn.BatchNorm1d(inception_inchannel)
		self.maxpool = torch.nn.MaxPool1d(kernel_size=3)
		self.dropout = torch.nn.Dropout(0.4)
		self.relu = torch.nn.ReLU(inplace=True)
	def forward(self, inputs, embed_gradient=None, debug=False):

		if debug:
			print(f'input_shape: {inputs.shape}')
		embedded = self.embedding(inputs.long())
		if debug:
			print(f'embedded: {embedded}')
		if embed_gradient is not None:
			embedded = torch.add(embedded, embed_gradient)
		if debug:
			print(f'embedded_shape: {embedded.shape}')
		conv1d_out = self.dropout(self.relu(self.batch_norm(self.conv1d(embedded))))
		if debug:
			print(f'conv1d_out_shape: {conv1d_out[-1]}')
		pool = self.maxpool(conv1d_out)
		incept_1_out = self.dropout(self.relu(self.inception(pool)))
		if debug:
			print(f'incept_1_out_shape: {incept_1_out[-1]}')
		pool = self.maxpool(incept_1_out)
		incept_2_out = self.dropout(self.relu(self.inception(pool)))
		if debug:
			print(f'incept_2_out_shape: {incept_2_out[-1]}')
		pool = self.maxpool(incept_2_out)
		## reshape befor fully connected layer
		pool = pool.view(pool.size(0), -1)
		if debug:
			print(f'pool_shape (before passed to linear): {pool.shape}')
		linear_out = self.linear(pool)
		if debug:
			print(f'linear_out_shape: {linear_out.shape}')
		# linear_out = self.softmax(linear_out)
		embedded_gradient = torch.ge(embedded.data, 0)
		if debug:
			print(f'embedded_gradient: {embedded_gradient}')
			print(f'embedded shape: {embedded.shape}')
			print(f'embedded_gradient shape: {embedded_gradient.shape}')
		return linear_out, embedded




class OOD_Classifier(torch.nn.Module):
	def __init__(self,hparams, words_count):
		super(OOD_Classifier, self).__init__()
		self.hparams = hparams
		# encoder = Encoder(hparams=hparams)
		# decoder = Decoder(hparams=hparams)
		# self.embedding = torch.nn.Embedding(num_embeddings=words_count+10, embedding_dim=512)

		self.generator = Generator()
		self.discriminator = Discriminator()
		self.auxiliary = Auxiliary_Classifier(hparams=hparams)
		self.autoencoder = Unilabel_Classifier(hparams=hparams, words_count=words_count)
		self.classifier = torch.nn.Sequential(*[self.autoencoder, self.auxiliary])
	def calculate_gradient_penalty(self, real_images, fake_images):
		if torch.cuda.is_available():
			eta = torch.cuda.FloatTensor(real_images.size(0), 1,).uniform_(0, 1)
		else:
			eta = torch.FloatTensor(real_images.size(0), 1, ).uniform_(0, 1)

		eta = eta.expand(real_images.size(0), real_images.size(1))
		interpolated = eta * real_images + ((1 - eta) * fake_images)
		# define it to calculate gradient
		interpolated = torch.tensor(interpolated, requires_grad=True, device=device)

		# calculate probability of interpolated examples
		prob_interpolated = self.discriminator(interpolated)

		# calculate gradients of probabilities with respect to examples
		gradients = torch.autograd.grad(outputs=prob_interpolated,
		                                inputs=interpolated,
		                                grad_outputs=torch.ones(prob_interpolated.size(), device=device),
		                                create_graph=True,
		                                retain_graph=True)[0]

		grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 ### lambda_term =10
		return grad_penalty
	# def train_generator(self, inputs, debug=False):
	# 	if debug:
	# 		print(f'inputs {inputs}')
	# 		print(f'self.generator {self.generator}')
	# 	g_out = self.generator(inputs)
	# 	return g_out
	#
	# def train_discriminator(self,inputs, debug=False):
	# 	if debug:
	# 		print(f'inputs {inputs}')
	# 		print(f'self.discriminator {self.discriminator}')
	# 	d_out = self.discriminator(inputs)
	# 	return d_out
	#
	# def train_auxiliary_classifier(self, inputs, debug=False):
	# 	if debug:
	# 		print(f'inputs {inputs.shape}')
	# 		print(f'self.auxiliary {self.auxiliary}')
	# 	aux_out = self.auxiliary(inputs)
	# 	return aux_out
	#
	# def train_autoencoder(self, inputs, debug=False):
	#
	# 	if debug:
	# 		print(f'inputs {inputs.shape}')
	# 		print(f'self.embedding {self.embedding}')
	# 		print(f'self.autoencoder {self.autoencoder}')
	# 	embedded = self.embedding(inputs)
	# 	auto_encoder_out, latten_code = self.autoencoder(embedded)
	# 	return auto_encoder_out.squeeze(-1), latten_code