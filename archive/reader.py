from email.mime import image
import numpy as np
import cv2
import imutils
import datetime







def capture_reading(x):
    #print (x)
    pass




def img_compare(img1, img2):
   diff = cv2.subtract(img1, img2)
   h, w = img1.shape
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   msre = np.sqrt(mse)
   return msre 

def proc_none(image) : 
  return image


def proc_0(image) : 

  r_low = cv2.getTrackbarPos('R_Low','control')
  g_low = cv2.getTrackbarPos('G_Low','control')
  b_low = cv2.getTrackbarPos('B_Low','control')

  r_high = cv2.getTrackbarPos('R_High','control')
  g_high = cv2.getTrackbarPos('G_High','control')
  b_high = cv2.getTrackbarPos('B_High','control')



  h_low = cv2.getTrackbarPos('H_Low','control')
  s_low = cv2.getTrackbarPos('S_Low','control')
  v_low = cv2.getTrackbarPos('V_Low','control')

  h_high = cv2.getTrackbarPos('H_High','control')
  s_high = cv2.getTrackbarPos('S_High','control')
  v_high = cv2.getTrackbarPos('V_High','control')


  
  rgb_lower = np.array([b_low, r_low, g_low])
  rgb_upper = np.array([r_high,g_high, b_high])
  rgb_mask = cv2.inRange(image, rgb_lower, rgb_upper)

  rgb_output = cv2.bitwise_and(image,image, mask= rgb_mask)

  # Create HSV Image and threshold into a range.
  hsv_lower = np.array([h_low, s_low, v_low])
  hsv_upper = np.array([h_high, s_high, v_high])
  hsv = cv2.cvtColor(rgb_output, cv2.COLOR_BGR2HSV)

  hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
  hsv_output = cv2.bitwise_and(rgb_output,rgb_output, mask= hsv_mask)
  output = hsv_output


  image_grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
  ret, image_thresh = cv2.threshold(image_grey, 127, 255, 0)
  contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.RETR_FLOODFILL)

  for c in contours:
    area = cv2.contourArea(c)
    if area > 400: 
      approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
      cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)


  return image


def proc_1(image) :
  #image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # define range of black color in HSV
  
  lower_val = np.array([0,0,0])
  upper_val = np.array([179,100,130])

  mask = cv2.inRange(image, lower_val, upper_val)
  result = cv2.bitwise_and(image,image, mask= mask)

  return result

def proc_2(image) :
  #image = proc_1(np.copy(image))
  image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ret, image_thresh = cv2.threshold(image_grey, 127, 255, 0)
  contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  for c in contours:
    area = cv2.contourArea(c)
    if area > 400: 
      approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
      cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)
  return image

def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("lcd_screen_win", cv2.WINDOW_NORMAL)
    cv2.namedWindow("control", cv2.WINDOW_NORMAL)

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

    cv2.createTrackbar('R_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('R_High', 'control', 0, 255, capture_reading)
    
    cv2.createTrackbar('G_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('G_High', 'control', 0, 255, capture_reading)

    cv2.createTrackbar('B_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('B_High', 'control', 0, 255, capture_reading)

    cv2.createTrackbar('H_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('H_High', 'control', 0, 255, capture_reading)
    
    cv2.createTrackbar('S_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('S_High', 'control', 0, 255, capture_reading)
    
    cv2.createTrackbar('V_Low', 'control', 0, 255, capture_reading)
    cv2.createTrackbar('V_High', 'control', 0, 255, capture_reading)
   


    # main loop: retrieves and displays a frame from the camera

    processor = proc_none
    prev_lcd_screen = None

    while (True):
        # blocks until the entire frame is read
        success, image = cap.read()

        # compute fps: current_time - last_time
        frames += 1
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)

        # wait 1ms for ESC to be pressed
        image_show = image

        # draw FPS text and display image
        #cv2.putText(image_show, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.rectangle(image_show, (225, 165), (425, 315), (0, 255, 0), 3)

        lcd_screen = image_show[165:315,225:425]
        lcd_screen = cv2.cvtColor(lcd_screen, cv2.COLOR_BGR2GRAY)
        
        pressure_high_digit_3 = lcd_screen[19:19 + 35, 30:30 + 58]
        pressure_high_digit_2 = lcd_screen[60:60 + 35, 30:30 + 58]
        pressure_high_digit_1 = lcd_screen[99:99 + 35, 30:30 + 58]

        pressure_low_digit_3 = lcd_screen[19:19 + 35, 93:93 + 58]
        pressure_low_digit_2 = lcd_screen[60:60 + 35, 93:93 + 58]
        pressure_low_digit_1 = lcd_screen[99:99 + 35, 93:93 + 58]


        heart_beat_digit_3   = lcd_screen[20:20 + 22, 153:153 + 36]
        heart_beat_digit_2   = lcd_screen[43:43 + 22, 153:153 + 36]
        heart_beat_digit_1   = lcd_screen[65:65 + 12, 153:153 + 36]

        heart_sign           = lcd_screen[77:77 + 20, 159:159 + 20]
        
        if prev_lcd_screen is None: prev_lcd_screen = lcd_screen
        if img_compare(prev_lcd_screen, lcd_screen) > 1 :
          cv2.imwrite("R:/temp2/LCD." + str(frames)  + ".jpg", lcd_screen)
          cv2.imwrite("R:/temp2/PH_1." + str(frames) + ".jpg", pressure_high_digit_1)
          cv2.imwrite("R:/temp2/PH_2." + str(frames) + ".jpg", pressure_high_digit_2)
          cv2.imwrite("R:/temp2/PH_3." + str(frames) + ".jpg", pressure_high_digit_3)

          cv2.imwrite("R:/temp2/PL_1." + str(frames) + ".jpg", pressure_low_digit_1)
          cv2.imwrite("R:/temp2/PL_2." + str(frames) + ".jpg", pressure_low_digit_2)
          cv2.imwrite("R:/temp2/PL_3." + str(frames) + ".jpg", pressure_low_digit_3)

          cv2.imwrite("R:/temp2/HB_1." + str(frames) + ".jpg", heart_beat_digit_1)
          cv2.imwrite("R:/temp2/HB_2." + str(frames) + ".jpg", heart_beat_digit_2)
          cv2.imwrite("R:/temp2/HB_3." + str(frames) + ".jpg", heart_beat_digit_3)
          cv2.imwrite("R:/temp2/HS."   + str(frames) + ".jpg", heart_sign)

        prev_lcd_screen = lcd_screen

        cv2.imshow("webcam", lcd_screen)
        cv2.imshow("lcd_screen_win", pressure_high_digit_1)


        key = cv2.waitKey(1)
        if (key == 27 or key==113 or key==81 ):
          break
        elif key == ord('1') :
          processor = proc_1
        elif key == ord('2') :
          processor = proc_2
        elif key == ord('3') :
          break
        elif key == ord('4') :
          break
        elif key == ord('5') :
          break
        elif key == ord('6') :
          break
        elif key == ord('7') :
          break
        elif key == ord('8') :
          break
        elif key == ord('9') :
          break
        elif key == ord('0') :
          processor = proc_none






    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()    