import time
import os
import random as ran

import numpy as np
import tensorflow as tf
import pygame.font, pygame.event, pygame.draw


# centers the popup window
os.environ['SDL_VIDEO_CENTERED'] = '1'

pygame.init()

black = (0, 0, 0)
gray = (200, 200, 215)
blue_gray = (45, 45, 60)
white = (255, 255, 255)
orange = (255, 128, 10)
bright_orange = (255, 170, 50)
green = (40, 255, 15)

width, height = 914, 612
size = [width, height]
half_width, half_height = width/2, height/2
input_field = (392, 392)
edge_buffer = (10, 50)
scale_size = 17.6

x_coordinates =[]
y_coordinates = []
pixel_colors = []

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Convolutional Neural Network")

loading_background = pygame.Surface(size)
loading_background.fill(gray)
backdrop = pygame.Surface(size)
backdrop.fill(gray)
background = pygame.Surface(input_field)
background.fill(white)
background2 = pygame.Surface((492, 492))
background2.fill(white)

font = pygame.font.SysFont("Agency FB", 40)
font_small = pygame.font.SysFont("Agency FB", 32)

message = "initializing convolutional neural network..."
bar =  "===================" #"<><><><><><><><><><><><>"
loadscreen = [bar, message]
clock = pygame.time.Clock()

def calculate_image(background):
    """ transforms the image into an array ready for matmult """

    scaledBackground = pygame.transform.smoothscale(background, (28, 28))
    image = pygame.surfarray.array3d(scaledBackground)
    image = abs(1-image/253)
    image = np.mean(image, 2)

    pixelate(image)

    image = image.transpose()
    image = image.ravel()
    return image

def calculate_prediction(input_draw):
    """ runs the transformed image through model to get a prediction """

    reshaped_input = input_draw.reshape(1,784)

    # feed the input into the trained model and output 10 predictions
    calculate = sess.run(y_, feed_dict={x: reshaped_input})
    prediction = calculate.argmax()

    # create list of probabilities for each number based off of one-hot encoding
    one_hot_probs = calculate[0]

    # call the predicted number's probability and mult by 100, then round to 2 decimals
    prob = round(one_hot_probs[prediction] * 100, 2)

    display_prediction(prediction, prob)

def display_prediction(prediction, prob):
    """ displays the prediction and probability on screen """

    display_prediction = "Prediction: %s" %(prediction)
    display_probability = "Probability: %s" %(prob)
    font = pygame.font.SysFont("Agency FB", 36)

    # converts to pygame format
    initialize_prediction = font.render(display_prediction, 1, (white))
    initialize_probability = font.render(display_probability + "%", 1, (white))

    # draws to screen
    pygame.draw.rect(screen, orange, (edge_buffer[0], input_field[1] + edge_buffer[1] + 10, input_field[0], 90))
    screen.blit(initialize_prediction, (edge_buffer[0] + 5, input_field[1] + edge_buffer[1] + 20))
    screen.blit(initialize_probability, (edge_buffer[0] + 5, input_field[1] + edge_buffer[1] + 60))

def pixelate(image):
    """ pixelates image """

    size = 28
    image = image.ravel()

    # creates RGB values for each pixel
    image = (255-image*255)

    # draws rect for each pixel
    for column in range(size):
        for row in range(size):
            # 0 - size**2
            index = row*size + column
            base_rgb = int(image[index])

            x_coordinates.append(row)
            y_coordinates.append(column)
            pixel_colors.append(base_rgb)

def draw_gradient():
    """ draws a gradient background """

    gradient = [230, 230, 255]
    x = 0

    for i in range(int(size[0]/4 + 1)):

        pygame.draw.rect(screen, gradient, (x, 0, 4, size[1]))
        gradient[0] -= 0.55
        gradient[1] -= 0.55
        gradient[2] -= 0.55
        x += 4

    pygame.display.flip()

def draw_loadscreen():
    """ draws main interface of the load screen """

    draw_gradient()

    # convert strings to pygame format
    initialize_loadscreen_bar = font_small.render(loadscreen[0], 1, black)

    initialize_loadscreen_message = font_small.render(loadscreen[1], 1, black)

    # stores width and height of features in lists
    loadscreen_bar_dim = initialize_loadscreen_bar.get_rect().width, initialize_loadscreen_bar.get_rect().height
    loadscreen_message_dim = initialize_loadscreen_message.get_rect().width, initialize_loadscreen_message.get_rect().height

    # adds features to screen and adjusted to center
    screen.blit(initialize_loadscreen_message, (half_width - loadscreen_message_dim[0]/2, half_height - loadscreen_message_dim[1]/2 - 55))
    screen.blit(initialize_loadscreen_bar, (half_width - loadscreen_bar_dim[0]/2, half_height - loadscreen_bar_dim[1]/2 - 32))
    screen.blit(initialize_loadscreen_bar, (half_width - loadscreen_bar_dim[0]/2, half_height - loadscreen_bar_dim[1]/2 + 15))

    # outlining rect
    pygame.draw.rect(screen, black, (half_width - 115, half_height - 22, 230, 25), 1)
    pygame.display.flip()

def loader():
    """ displays loading bar on screen """

    loader = ""

    for i in range(35):

        timer = ran.uniform(0.03, 0.1)
        loader += "/"

        initialize_loader = font_small.render(loader, 1, orange)
        loadscreen_loader_dim = initialize_loader.get_rect().width, initialize_loader.get_rect().height

        # redraw screen to prevent overlap of loader
        draw_loadscreen()
        screen.blit(initialize_loader, (half_width - 105, half_height - 21))

        # update screen
        pygame.display.flip()
        time.sleep(timer)

    screen.blit(backdrop, (0, 0))
    pygame.display.flip()

def scanner():
    """ creates visual scanner for user input """

    changeY = 0
    speed = 0.003

    # used to call values that stores data for each pixel
    coordinate_x = 1
    coordinate_y = 1
    px = 1

    for x in range(int(input_field[0]/2)):

        # draw background each time to overdraw scanner
        screen.blit(background,(edge_buffer))
        pygame.draw.rect(screen, green, (edge_buffer[0], edge_buffer[1] + changeY, input_field[0], 5))
        changeY += 2

        # every 7 iterations, draws line of pixelated image
        if changeY % 14 == 0:

            # staggers the pixelated image
            if (changeY/14) % 2 == 0:

                coordinate_x += 1
                coordinate_y += 1
                px += 1

            else:

                coordinate_x -= 1
                coordinate_y -= 1
                px -= 1

            for i in range(14):

                gray_scaled = (pixel_colors[px], pixel_colors[px], pixel_colors[px])
                pygame.draw.rect(screen, gray_scaled, (x_coordinates[coordinate_x]*scale_size + input_field[0] + 2*edge_buffer[0],
                y_coordinates[coordinate_y]*scale_size + edge_buffer[1], scale_size, scale_size))

                coordinate_x += 2
                coordinate_y += 2
                px += 2

        # draws rectange same as background to compensate for overlap of scanner
        pygame.draw.rect(screen, gray, (edge_buffer[0], edge_buffer[1] + input_field[1], input_field[0], 10))\

        # update the screen
        time.sleep(speed)
        pygame.display.flip()
    # goes from down to up
    coordinate_x = 783
    coordinate_y = 783
    px = 783

    for x in range(int(input_field[0]/2)):

        screen.blit(background,(edge_buffer))
        pygame.draw.rect(screen, green, (edge_buffer[0], edge_buffer[1] + changeY, input_field[0], 5))
        changeY -= 2

        # every 7 iterations, draws line of pixelated image
        if changeY % 14 == 0:

            # staggers the pixelated image, opposite of the first scanner
            if (changeY/14) % 2 == 0:
                coordinate_x += 1
                coordinate_y += 1
                px += 1

            else:
                coordinate_x -= 1
                coordinate_y -= 1
                px -= 1

            # fills in remaining pixels on way up
            for i in range(14):
                gray_scaled = (pixel_colors[px], pixel_colors[px], pixel_colors[px])
                pygame.draw.rect(screen, gray_scaled, (x_coordinates[coordinate_x]*scale_size + input_field[0] + 2*edge_buffer[0],
                y_coordinates[coordinate_y]*scale_size + edge_buffer[1], scale_size, scale_size))
                coordinate_x -= 2
                coordinate_y -= 2
                px -= 2
        # prevents scanner from going past background and staying on screen
        pygame.draw.rect(screen, gray, (edge_buffer[0], edge_buffer[1] + input_field[1], input_field[0], 10))

        # update screen at a set pace
        time.sleep(speed)
        pygame.display.flip()

    # clear lists holding pixel data
    x_coordinates[:] = []
    y_coordinates[:] = []
    pixel_colors[:] = []

def create_button(btn_label, surface, color, new_color, locationX, locationY, width, height):
    """ creates interactive classify btn """

    # convert label to pygame format
    initialize_btn_label = font_small.render(btn_label, 1, white)
    initialize_btn_label_dim = initialize_btn_label.get_rect().width, initialize_btn_label.get_rect().height
    pygame.draw.rect(surface, color, (locationX, locationY, width, height))
    mouse = pygame.mouse.get_pos()

    # checks if mouse is with in boundaries of buttton and updates color
    if locationX + width > mouse[0] > locationX and locationY + height> mouse[1] > locationY:
        pygame.draw.rect(surface, new_color, (locationX, locationY, width, height))

        # if classify btn is clicked
        if pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Classify":
            image = calculate_image(background)
            scanner()
            calculate_prediction(image)

        # if clear btn is clicked
        elif pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Clear":

            # resets both canvases and predictions
            background.fill(white)
            screen.blit(background2, (2*edge_buffer[0] + input_field[0], edge_buffer[1]))
            display_prediction('Unknown', "0")

    else:
        pygame.draw.rect(surface, color, (locationX, locationY, width, height))

    # adds the btn label to the btn
    surface.blit(initialize_btn_label, (locationX + width/2 - initialize_btn_label_dim[0]/2,
        locationY + height/2 - initialize_btn_label_dim[1]/2))

def draw_line(surface, color, start, end, radius):
    """ draws a line """

    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start[0]+i/distance*dx)
        y = int(start[1]+i/distance*dy)
        pygame.draw.circle(surface, color, (x - edge_buffer[0], y - edge_buffer[1]), radius)

def draw_interface():
    """ draws main components of interface """

    loader()
    display_prediction('Unknown', "0")

    label_input = "Input"
    label_pixelated = "Pixelated"
    label_mnist = "Model trained on the MNIST database"

    # convert to pygame format
    initialize_label_input = font.render(label_input, 1, black)
    initialize_label_pixelated = font.render(label_pixelated, 1, black)
    initialize_label_mnist = font.render(label_mnist, 1, white)

    # add to screen
    screen.blit(initialize_label_input, (edge_buffer[0] + 10, 10))
    screen.blit(initialize_label_pixelated, (input_field[0] + 2*edge_buffer[0] + 10, 10))
    pygame.draw.rect(screen, blue_gray, (edge_buffer[0], size[1] - 60, size[0] - 2*edge_buffer[0], 50))
    screen.blit(initialize_label_mnist, (size[0]/2 - initialize_label_mnist.get_rect().width/2, size[1] - 50))
    screen.blit(background2, (2*edge_buffer[0] + input_field[0], edge_buffer[1]))

    # update screen
    pygame.display.flip()

def main():
    """ draws interface """

    last_pos = (0, 0)
    line_width = 9

    draw_interface()

    image = None
    continue_on = True
    while continue_on:
        for event in pygame.event.get():

            # stops program when window closed
            if event.type == pygame.QUIT:
                continue_on = False

            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    draw_line(background, black, event.pos, last_pos, line_width)
                last_pos = event.pos

            # update screen
            screen.blit(background, (edge_buffer[0], edge_buffer[1]))
            create_button("Classify", screen, orange, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 10, 120, 45)
            create_button("Clear", screen, orange, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 55, 120, 45)
            pygame.display.flip()

""" the cnn model below is not my work """
""" tutorial teaching how to use it: http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/ """
""" source code: https://github.com/adventuresinML/adventures-in-ml-code """

x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
# -1 placeholder, tf will reshape based on number of training examples
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

#y = tf.placeholder(tf.float32, [None, 10])

def conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):

    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,

                      num_filters]

    # initialise weights and bias for the filter
    W = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, W, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer

layer1 = conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.nn.relu(tf.matmul(flattened, wd1) + bd1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

""" end of cnn model """


saver = tf.train.Saver()
sess = tf.Session()

with sess as sess:
    saver.restore(sess, "data/convolutional.ckpt")
    main()
