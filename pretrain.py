import os
import tensorflow as tf
from BertLayer import Bert
from Data.data import DataGenerator
from Loss.loss import BERT_Loss
from Loss.utils import calculate_pretrain_task_accuracy
from config import Config
from datetime import datetime

## HELPER FUNCTIONS

def compute_loss(per_example_loss, global_batch_size):
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


def train_step(batch_x, batch_padding_mask, batch_segment):
    global nsp_predict, mlm_predict
    with tf.GradientTape() as t:
        nsp_predict, mlm_predict, sequence_output = model((batch_x, batch_padding_mask, batch_segment), training=True)
        nsp_loss, mlm_loss = loss_fn((mlm_predict, batch_mlm_mask, origin_x, nsp_predict, batch_y))
        # nsp_loss = tf.reduce_mean(nsp_loss)
        # mlm_loss = tf.reduce_mean(mlm_loss)
        loss = nsp_loss + mlm_loss
        loss = compute_loss(loss, global_batch_size) # replace the above two lines with this one

    gradients = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def distributed_train_step(batch_x, batch_padding_mask, batch_segment):
    per_replica_losses = mirrored_strategy.run(train_step, args=(batch_x, batch_padding_mask, batch_segment))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

## MAIN FUNCION

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

mirrored_strategy = tf.distribute.MirroredStrategy()
global_batch_size = Config["Batch_Size"]

# initialize optimizer using mirrored strategy
with mirrored_strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model = Bert(Config)
    loss_fn = BERT_Loss()

# distribute dataset onto multiple GPUs
dataset = DataGenerator(Config)
# dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(Config['Saved_Weight']))
manager = tf.train.CheckpointManager(checkpoint, directory=Config['Saved_Weight'], max_to_keep=5)
log_dir = os.path.join(Config['Log_Dir'], datetime.now().strftime("%Y-%m-%d"))
writer = tf.summary.create_file_writer(log_dir)


EPOCH = 10
for epoch in range(EPOCH):
    for step in range(len(dataset)):
        batch_x, batch_mlm_mask, origin_x, batch_segment, batch_padding_mask, batch_y = dataset[step]

        loss = distributed_train_step(batch_x, batch_padding_mask, batch_segment)
        nsp_acc, mlm_acc = calculate_pretrain_task_accuracy(nsp_predict, mlm_predict, batch_mlm_mask, origin_x, batch_y)

        with writer.as_default():
            global_step = epoch * len(dataset) + step
            tf.summary.scalar('train_loss', loss.numpy(), step=global_step)
            tf.summary.scalar('nsp_accuracy', nsp_acc, step=global_step)
            tf.summary.scalar('mlm_accuracy', mlm_acc, step=global_step)

    path = manager.save(checkpoint_number=epoch)
    print('model saved to %s' % path)
