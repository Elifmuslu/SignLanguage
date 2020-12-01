import cv2
import time
import numpy as np
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def nothing(x):
    pass


image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('C:/Users/ITU/PycharmProjects/İşaretDiliAlgılamaProjesi/mydata/training_set/' + folder_name):
        os.mkdir('C:/Users/ITU/PycharmProjects/İşaretDiliAlgılamaProjesi/mydata/training_set/' + folder_name)
    if not os.path.exists('C:/Users/ITU/PycharmProjects/İşaretDiliAlgılamaProjesi/mydata/test_set/' + folder_name):
        os.mkdir('C:/Users/ITU/PycharmProjects/İşaretDiliAlgılamaProjesi/mydata/test_set/' + folder_name)
    
        

        
def capture_images(ges_name):
    create_folder(str(ges_name))
    
    cam = cv2.VideoCapture(0) #Capture the video stream from default or supplied capturing device.

    cv2.namedWindow("test") #Create a window to display the default frame and the threshold frame.

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1,2,3,4,5]

    cv2.namedWindow("Trackbars") #Create the trackbars to set the range of HSV values

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)  #lower blue: [0,0,0]
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing) #upper blue:[179,255,255]
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    for loop in listImage:
        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")  #HSV renk uzayını kullanmak daha elverişlidir. Çünkü RGB'nin
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")  #aksine sadece hue değerini kullanarak eşik değer uygulama
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")  #suretiyle renkleri daha net ayırt edebiliriz.
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            imcrop = img[102:298, 427:623]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV) #we have convert the image to an hsv image because hsv helps to differentiate intensity from color.
            mask = cv2.inRange(hsv, lower_blue, upper_blue) #we define the upper and lower limit of the blue we want to detect.
            #Here we are actually creating a mask with the specified blue. The mask simply represent a specific part of the image.
            #İn this case, we are checking through the hsv image, and checking for colors that are between the lower-range and upper-range.
            #The areas that match will an image set to the mask variable.
            result = cv2.bitwise_and(imcrop, imcrop, mask=mask)
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("test", frame) #Show the images
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) == ord('c'):

                if t_counter <= 350:
                    img_name = "./mydata/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1


                if t_counter > 350 and t_counter <= 400:
                    img_name = "./mydata/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 250:
                        break


                t_counter += 1
                if t_counter == 401:
                    t_counter = 1
                img_counter += 1


            elif cv2.waitKey(1) == 27: ## wait for ESC key to exit
                break

        if test_set_image_name > 250:
            break


    cam.release()
    cv2.destroyAllWindows()
    
ges_name = input("Enter gesture name: ")
capture_images(ges_name)