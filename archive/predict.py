import os, sys
import cv2
import py3toolbox as tb
import tensorflow as tf
import numpy as np
import random

CONFIG_FILE = 'config.json'
class_names = ['0','1','2','3','4','5','6','7','8','9','BLANK','NA']

def get_config():
  config = tb.load_json(CONFIG_FILE)
  return config


def load_model() :
  config = get_config()
  model = tf.keras.models.load_model(config['MODEL_NAME'])
  model.load_weights(config['CHECKPOINT_DIR'])
  print (model.summary())
  #checkpoints = tf.compat.v1.train.checkpoint_management.list_all_checkpoints(config['CHECKPOINT_DIR'])
  #print(checkpoints)
  return model





def main():
  config = get_config()
  model = load_model()
  print (model.summary())
  tb.pause()
  files = []
  for f in tb.gen_files('D:/Projects/blood_pressure_reader/labeled_images/BIG'):
    files.append(f)

  for i in range(60) :
    f = random.choice(files)
    image = cv2.imread(f)
    final_image = np.reshape(image, (1, image.shape[0], image.shape[1],3))
    predictions = model.predict(final_image)
    print ("===", class_names[np.argmax(predictions)])
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
  


if __name__ == "__main__" :
  tf.config.list_physical_devices('GPU')
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  main()