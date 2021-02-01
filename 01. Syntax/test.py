from pathlib import Path

training_path =Path("/Users/Tyler/PycharmProjects/OCR/document quality/LBP training/images/training")
testing_path =Path("/Users/Tyler/PycharmProjects/OCR/document quality/LBP training/images/testing")
filename = 'finalized_model.sav'
train = 1
if train == True:
    data = []
    labels = []

    # loop over the training images
    # sum = paths.list_images(training_path).
    count = 0
    image_list = [str(i) for i in list(training_path.glob("**/*.jpg"))]
    print(image_list)