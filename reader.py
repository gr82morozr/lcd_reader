
import numpy as np
import cv2
import datetime
import py3toolbox as tb
import tensorflow as tf
import numpy as np

CONFIG_FILE = '.\config.json'

def get_config():
  config = tb.load_json(CONFIG_FILE)
  return config


def load_model() :
  config = get_config()
  model = tf.keras.models.load_model(config['MODEL_NAME'])
  model.load_weights(config['CHECKPOINT_DIR'])
  #print (model.summary())
  return model

def img_compare(img1, img2):
   diff = cv2.subtract(img1, img2)
   h, w = img1.shape
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   msre = np.sqrt(mse)
   return msre 

def proc_none(image) : 
  return image

def read_digit (model, image):
  config = get_config()
  class_names = config["CLASS_NAMES"]
  final_image = np.reshape(image, (1, image.shape[0], image.shape[1],3))
  predictions = model.predict(final_image, verbose = 0)
  result = class_names[np.argmax(predictions)]
  if result == 'BLANK' or result == 'NA' : result = 0
  return (int(result))

def main():
  config = get_config()
  model = load_model()

  # create display window
  cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

  # initialize webcam capture object
  cap = cv2.VideoCapture(0)

  # retrieve properties of the capture object
  cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  cap_fps = cap.get(cv2.CAP_PROP_FPS)
  fps_sleep = int(1000 / cap_fps)
  print('* Capture width:', cap_width)
  print('* Capture height:', cap_height)
  print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

  # initialize time and frame count variables
  last_time = datetime.datetime.now()
  frames = 0

  pressure_high_prev = 0
  pressure_low_prev  = 0
  stable_count        = -1 # for auto detection of the measurement cycle
  cycle = 0
  while (True):
    # blocks until the entire frame is read
    success, image = cap.read()

    # compute fps: current_time - last_time
    frames += 1
    delta_time = datetime.datetime.now() - last_time
    elapsed_time = delta_time.total_seconds()
    cur_fps = np.around(frames / elapsed_time, 1)

    # Locate the digits from LCD screen
    lcd_screen = image[165:315,225:425]
    pressure_high_digit_3 = lcd_screen[19:19 + 35, 30:30 + 58]
    pressure_high_digit_2 = lcd_screen[60:60 + 35, 30:30 + 58]
    pressure_high_digit_1 = lcd_screen[99:99 + 35, 30:30 + 58]
    pressure_low_digit_3 = lcd_screen[19:19 + 35, 93:93 + 58]
    pressure_low_digit_2 = lcd_screen[60:60 + 35, 93:93 + 58]
    pressure_low_digit_1 = lcd_screen[99:99 + 35, 93:93 + 58]

    pressure_high = read_digit(model, pressure_high_digit_1) * 100 + read_digit(model, pressure_high_digit_2) * 10 + read_digit(model, pressure_high_digit_3)
    pressure_low  = read_digit(model, pressure_low_digit_1) * 100  + read_digit(model, pressure_low_digit_2) * 10  + read_digit(model, pressure_low_digit_3)
    print (stable_count, ":", cycle, "  :  " , pressure_high, " ==== ", pressure_low)

    # when start measuring ... (starting new measurement cycle)
    if stable_count == -1 and pressure_high > 0 and pressure_high_prev > 0 and pressure_high != pressure_high_prev and pressure_low == 0 :
      stable_count = 0
      print ("New cycle started.")

    # if reading unchanged for 6 frames
    if pressure_high_prev == pressure_high and pressure_low_prev  == pressure_low and pressure_low > 10 and stable_count > -1 :
      stable_count +=1
    elif stable_count == -1 :
      stable_count = -1
    else:
      stable_count = 0
      
    if stable_count > 6:
      tb.write_file(file_name=config["LOG_FILE"], text=tb.get_timestamp()+ "," + str(pressure_high) + "," + str(pressure_low) + "\n", mode="a")
      print ("Log written.")
      cycle  += 1
      stable_count = -1  # ready for new cycle

    pressure_high_prev = pressure_high
    pressure_low_prev  = pressure_low

    cv2.imshow("webcam", lcd_screen)

    key = cv2.waitKey(1)
    if (key == 27 or key==113 or key==81 ):
      break


  # release resources
  cv2.destroyAllWindows()
  cap.release()


if __name__ == "__main__":
    main()    