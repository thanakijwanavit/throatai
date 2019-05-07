#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER:  nic wanavit
# DATE CREATED: 26 oct 18
# REVISED DATE:
# PURPOSE: Create a function that retrieves the following 3 command line inputs
#          from the user using the Argparse Python module. If the user fails to
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument
#       collection that you created with this function
#
def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 3 command line arguments. If
    the user fails to provide some or all of the 3 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: that's a path to a folder
    parser.add_argument('--save_dir', type = str, default = 'saved_classifier/checkpoint.pth',
                    help = 'path to save the trained model')
    # Argument 2: CNN model Architecture
    parser.add_argument('--arch', type = str, default = 'vgg13', help = 'cnn model architecture')
    # Argument 3: text file with dog names
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'file with the dog names')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'file with the dog names')
    parser.add_argument('--epochs', type = int, default = 20, help = 'file with the dog names')
    parser.add_argument('image_dir', type = str, default ='flowers/valid/1/image_06739.jpg', help ='image directory')
    parser.add_argument('model_directory', type = str, default = 'saved_classifier/checkpoint.pth', help = 'location of the trained model')
    parser.add_argument('--gpu',action='store_true',help='process with GPU [default CPU]')
    parser.add_argument('--category_names', type = str, default ='None', help ='provide alternative category name')
    parser.add_argument('--top_k', type = int, default = 1, help = 'select the top k (default=1)')
    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()
