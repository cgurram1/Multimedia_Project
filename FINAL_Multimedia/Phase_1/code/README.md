Multimedia and Web Databases - CSE 515 - Phase 1
Project Structure:
The entire project is organized into three main parts: code, output, and a detailed report. The code section consists of eight Python files and a README.md file. Below is an explanation of the starter.py file and its functionality.

starter.py - Initialization and Execution File --> 
    This is the primary file to execute the entire project. The project is implemented in Python 3, and you can run it by executing the following command in the terminal: python3 starter.py

Upon execution, the user is presented with options on how to interact with the project:

1. Test: Choose this option if you have an image ID and want to display the top 10 most similar images to your input image using five different vector spaces (Color Moments, Histogram Of Gradients, Average Pool, Layer 3, and FC layers of ResNet50). Selecting this option will call the TestInteraction.py file, which displays all available vector spaces. You can then choose one from the list. This will internally invoke the Testing_images.py script, which generates the results and stores them in the "outputs" folder. The result will be a JPEG image containing the 10 most similar images to your input image.

2. Store: Use this option to populate the database (in my case, MySQL, local) with all the feature vectors. This option calls the Generator.py script to generate the feature vectors for all five vector spaces and stores them in the database. This database is used in the Test phase to retrieve the k-nearest images to the input image using various distance measures.

3. Generate: Choose this option if you want to view the feature vectors of any input image. This option allows you to select an image and a specific vector space. It then displays the feature vectors in a more human-readable format on the screen.