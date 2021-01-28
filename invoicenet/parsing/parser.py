import tensorflow as tf

from ..common.model import Model
from .parsers import DateParser, AmountParser
from .data import ParseData


class Parser(Model):

    def __init__(self, field, restore=False):
        self.type = field
        self.continue_from = './models/parsers/{}/best'.format(self.type) if restore else None

        self.parser = {'amount': AmountParser(), 'date': DateParser()}[self.type]

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)

        self.parser.compile(self.optimizer)

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.parser)

        if self.continue_from:
            print("Restoring " + self.continue_from + "...")
            self.checkpoint.read(self.continue_from)

    def loss_func(self, y_true, y_pred):
        mask = tf.logical_not(tf.equal(y_true, ParseData.pad_idx))
        label_cross_entropy = tf.reduce_mean(
            self.loss_object(y_true, y_pred) * tf.cast(mask, dtype=tf.float32)) / tf.math.log(2.)
        return label_cross_entropy

    @tf.function
    def train_step(self, inputs):
        inputs, targets = inputs

        with tf.GradientTape() as tape:
            predictions = self.parser(inputs, training=True)
            loss = self.loss_func(targets, predictions)

        gradients = tape.gradient(loss, self.parser.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.parser.trainable_variables))
        return loss

    @tf.function
    def val_step(self, inputs):
        inputs, targets = inputs
        predictions = self.parser(inputs, training=False)
        loss = self.loss_func(targets, predictions)
        return loss

    def save(self, name):
        self.checkpoint.write(file_prefix="./models/parsers/%s/%s" % (self.type, name))

    def load(self, name):
        self.checkpoint.read(name)
