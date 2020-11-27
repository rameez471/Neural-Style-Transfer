import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from utils import vgg_layers, gram_matrix, clip
import time

#List of layer for Content 
content_layers = ['block5_conv2']
#List of layer for styling
style_layers = ['block1_conv1',
               'block2_conv1',
               'block3_conv1',
               'block4_conv1',
               'block5_conv1']

class StyleContentModel(Model):

    def __init__(self,style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg  = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self,inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocessed_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name:value for content_name,value in zip(self.content_layers,content_outputs)}
        
        style_dict = {style_name:value for style_name,value in zip(self.style_layers,style_outputs)}

        return {'content':content_dict,
                'style':style_dict}


class TransferModel:

    def __init__(self, style_image, content_image):
        self.extractor = StyleContentModel(style_layers, content_layers)
        self.style_targets = self.extractor(style_image)['style']
        self.content_targets = self.extractor(content_image)['content']
        self.image = tf.Variable(content_image)
        self.style_weight = 1e-2
        self.content_weight = 1e4
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def style_content_loss(self,outputs):
        
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2)
                            for name in style_outputs.keys()])

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2)
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight/len(content_layers)
        
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self,image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip(image))


    def fit(self,epochs=10, steps_per_epoch=100):
        
        start = time.time()
        step = 0

        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(self.image)
                print('.',end='')
            print('Train Step: {}'.format(step))

        end = time.time()

        print('Total time: {:.1f}'.format(end - start))