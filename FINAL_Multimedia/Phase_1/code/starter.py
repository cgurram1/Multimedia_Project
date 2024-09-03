
import warnings
import torchvision
import TestInteraction
import GenerateInteraction
import Training

# load the caltech101 dataset to "dataset" variable
dataset = torchvision.datasets.Caltech101('/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/NewCaltech',download=False)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")

task = input(
"""Welcome to my project!. What do you want? (Select one from below)
    1. Test
    2. Store
    3. Generate
"""
    )
if(task == "1" or task == "3"):
    image_ID = (int(input("Enter the image ID ")))
    if(task == "1"):
        TestInteraction.Test(image_ID,dataset) # choose this if you have an image Id and want to test it with images in the database
    else:
        GenerateInteraction.Generate(image_ID,dataset) # choose this if you have an image Id and want to display the feature vectors
elif(task == "2"):
    Training.Train(dataset) # choose this if you want to store to the database, For now the data is already present in Final_Feature_Descriptor table
else:
    print("Invalid Selection. Please Select from the List")
    exit(0)