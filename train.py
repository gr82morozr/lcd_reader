import os, sys
import cv2
import py3toolbox as tb
import tensorflow as tf
import numpy as np

CONFIG_FILE = 'config.json'

def get_config():
  config = tb.load_json(CONFIG_FILE)
  return config


def prepare_data() :
  config = get_config()
  image_folder = config['IMAGE_FOLDER']
  labels = []
  image_height = 35
  image_width  = 58
  batch_size   = 1
  class_names = ['0','1','2','3','4','5','6','7','8','9','BLANK','NA']


  train_ds, val_ds = tf.keras.utils.image_dataset_from_directory (
    image_folder,
    labels = 'inferred',
    label_mode = "int",
    class_names = class_names,
    color_mode='rgb',
    batch_size = batch_size,
    image_size  = (image_height, image_width),
    shuffle = True,
    seed = 908,
    validation_split= 0.2,
    subset="both"
  )


  data = {}
  data["train"] = train_ds
  data["validation"] = val_ds

  # debug
  """
  for image, label in train_ds.take(10):
    image =  tf.squeeze(image)
    tf.print(label)
    print (image.shape)
    
    cv2.imshow('image', image.numpy().astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  """

  return (data)



def gen_model():
  model = tf.keras.Sequential([
    #tf.keras.layers.Rescaling(1./255, input_shape=(58, 35, 3)),
    tf.keras.layers.Conv2D(16, (3,2), padding='same', activation='relu', input_shape=(35, 58, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3,2), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),            
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,  activation='relu'),
    tf.keras.layers.Dense(12,   activation='softmax')
  ])
  return model


def train_model(model, data, epochs) :
  config = get_config()
  train_ds = data["train"]
  val_ds   = data["validation"]
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config['CHECKPOINT_DIR'], 
    save_best_only = True,
    save_weights_only=True, 
    verbose=1, 
    save_freq='epoch')
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
    callbacks=[cp_callback]
  )
  model.save(config['MODEL_NAME'])

def main():
  config = get_config()
  data = prepare_data()

  model = gen_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  print (model.summary())
  train_model(model, data, 80)

if __name__ == "__main__" :
  tf.config.list_physical_devices('GPU')
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  main()