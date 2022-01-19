import cv2

vidcap = cv2.VideoCapture('./videoData/six_people.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  if count % 2 == 0:
    cv2.imwrite("./picture/six_people/six_frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1