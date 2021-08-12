import tkinter
from tkinter.constants import DISABLED, NORMAL
import numpy
from keras.models import load_model
import PIL
import cv2
import os

class GUI(object):

    DEFAULT_COLOR = 'black'
    MODEL_FILE_NAME = 'MINST-Model.h5'

    def __init__(self):
        # import the model
        self.model = load_model(self.MODEL_FILE_NAME)

        # Start gui
        self.root = tkinter.Tk()

        # Reset button
        reset_button = tkinter.Button(self.root, text='Reset', command=self.clear)
        reset_button.grid(row=0, column=0)

        # Predict button
        predict_button = tkinter.Button(self.root, text='Predict', command=self.predict)
        predict_button.grid(row=0, column=1)

        # Drawing canvas
        self.old_x = None
        self.old_y = None
        self.canvas = tkinter.Canvas(self.root, bg='white', width=220, height=220)
        self.canvas.grid(row=1, columnspan=2, rowspan=11)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        
        # Prediction Value labels
        self.digit_percentages = []
        self.digit_labels = []
        tkinter.Label(self.root, text='Digit:').grid(row=1, column=3)
        tkinter.Label(self.root, text='Predicted Precentage:').grid(row=1, column=4)
        for i in range(10):
            digit_label = tkinter.Label(self.root, text=i)
            digit_label.grid(row=i+2, column=3)
            self.digit_labels.append(digit_label)

            digit_percentage = tkinter.Label(self.root, text='')
            digit_percentage.grid(row=i+2, column=4)
            self.digit_percentages.append(digit_percentage)

        # Reinforce the model
        tkinter.Label(self.root, text='Help improve the model: ').grid(row=12, columnspan=5)
        tkinter.Label(self.root, text='Choose the number you drew: ').grid(row=13, columnspan=2)
        # Drop down menu
        self.choice = tkinter.StringVar(self.root)
        self.choice.set(0)
        pop_up_menu = tkinter.OptionMenu(self.root, self.choice, *list(range(10)))
        pop_up_menu.grid(row=13, column=3)
        # Reinforce button
        self.train_button = tkinter.Button(self.root, text='Train', state=DISABLED, command=self.train)
        self.train_button.grid(row=13, column=4)

        # Start the GUI
        self.root.mainloop()

        # Code to save the model after the window has been closed 
        self.model.save(self.MODEL_FILE_NAME)
        
    # Function to predict the drawn value 
    def predict(self):
        # save canvas drawing
        self.canvas.postscript(file='digit.eps')
        temp_canvas_img = PIL.Image.open('digit.eps')
        temp_canvas_img.save('digit.png', 'png')

        # Reimport the PNG
        canvas_image = cv2.imread('digit.png')
        # Convery the image to black and white
        canvas_image = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
        # convert to array 
        canvas_image = numpy.asarray(canvas_image)   
        # resize to target shape            
        canvas_image = cv2.resize(canvas_image, (28, 28))   
        # [optional] my input was white bg, I turned it to black, so turns 1's into 0's and 0's into 1's
        canvas_image = cv2.bitwise_not(canvas_image)  
        # normalize (so the value is between 0-1 not 0-255) 
        canvas_image = canvas_image / 255  
        # Reshape the array to the required shape to be used by our model                  
        canvas_image = canvas_image.reshape(1, 28, 28, 1)

        self.train_image = canvas_image
        # self.train_image = numpy.array(canvas_image)

        # get the prediction of what digit
        percentages = self.model.predict(canvas_image)
        # Find the map proability and the equivalent digit it relates too
        predicted_digit = numpy.argmax(percentages,axis=1)[0]

        # delete temp saved images images
        if os.path.exists("digit.eps"):
            os.remove("digit.eps")
        if os.path.exists("digit.png"):
            os.remove("digit.png")
        
        # update GUI
        for i in range(10):
            self.digit_percentages[i]['text'] = percentages[0][i]
            # Highlight the predicted digit in red
            if i == predicted_digit:
                self.digit_percentages[i].config(fg='green')
                self.digit_labels[i].config(fg='green') 
        
        # disable the train button 
        self.train_button.config(state=NORMAL) 


    # Method to train the model
    def train(self):
        # disable the train button 
        self.train_button.config(state=DISABLED) 

        # set up label
        label = [0] * 10
        label[int(self.choice.get())] = 1
        label = [label]
        
        # Train the model
        self.model.fit(self.train_image, numpy.array(label), batch_size=1, epochs=1) 


    # to add the drawing to the canvas
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=18, fill=self.DEFAULT_COLOR,
                               capstyle=tkinter.ROUND, smooth=tkinter.TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    # Reset the brush so you are able to draw after clicking off
    def reset(self, event):
        self.old_x, self.old_y = None, None

    # reset the drawing canvas
    def clear(self):
        # Clear canvas
        self.canvas.delete('all')
        
        # Clear results
        for i in range(10):
            self.digit_percentages[i]['text'] = ''
            self.digit_percentages[i].config(fg=self.DEFAULT_COLOR)
            self.digit_labels[i].config(fg=self.DEFAULT_COLOR)
        
        # disable the train button 
        self.train_button.config(state=DISABLED) 

# Start the code
if __name__ == '__main__':
    GUI()