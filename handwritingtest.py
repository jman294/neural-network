from tkinter import *
import numpy
import matplotlib.pyplot as plot
from neuralnetwork import NeuralNetwork

input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = .2

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open('data/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 4

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255 * .99) + .01
        targets = numpy.zeros(output_nodes) + .01
        targets[int(all_values[0])] = .99
        nn.train(inputs, targets)
        pass
    pass
test_data_file = open('data/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(correct_label)

    inputs = (numpy.asfarray(all_values[1:]) / 255 * .99) + .01
    outputs = nn.query(inputs)
    label = numpy.argmax(outputs)
    print(label, '\n')

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

scorecard_array = numpy.asarray(scorecard)
print('perforamance = ', scorecard_array.sum() / scorecard_array.size)

# b1 = "up"
# xold, yold = None, None

# def main():
    # root = Tk()

    # buttonFrame = Frame(root)

    # b = Button(buttonFrame, text="Guess", command=guess)
    # b.pack()

    # buttonFrame.pack(side=BOTTOM)

    # canvasFrame = Frame(root)

    # drawing_area = Canvas(canvasFrame, width=28, height=28, highlightbackground="blue")
    # drawing_area.pack()
    # canvasFrame.pack(side=TOP)

    # drawing_area.bind("<Motion>", motion)
    # drawing_area.bind("<ButtonPress-1>", b1down)
    # drawing_area.bind("<ButtonRelease-1>", b1up)
    # root.resizable(width=False, height=False)
    # root.geometry('{}x{}'.format(90, 90))
    # root.mainloop()

# def guess():
    # print('guess')

# def b1down(event):
    # global b1
    # b1 = "down"           # you only want to draw when the button is down
                          # # because "Motion" events happen -all the time-

# def b1up(event):
    # global b1, xold, yold
    # b1 = "up"
    # xold = None           # reset the line when you let go of the button
    # yold = None

# def motion(event):
    # if b1 == "down":
        # global xold, yold
        # if xold is not None and yold is not None:
            # event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE)
    # # here's where you draw it. smooth. neat.
        # xold = event.x
        # yold = event.y

# main()
