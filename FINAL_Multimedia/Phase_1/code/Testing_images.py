import distances
import mysql.connector
import matplotlib.pyplot as plt
import Generator
import datetime

#creating the SQL connection
connection = mysql.connector.connect(host='localhost',username='root',password='GURRAMgckr@1998',database='Feature_Descriptors')

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

# These below functions take image_id(input image) and dataset as input
# Test_Color_Moments
# Test_HoG
# Test_AvgPool
# Test_layer_3
# Test_FC


def Test_Color_Moments(id,dataset):
    image = dataset[id][0] # store the image from the given image id using dataset variable
    if (image.mode not in ('RGB', 'RGBA', 'CMYK', 'YCbCr')):  #This is a color Moment vector space, so skipping all the grayscale images
        print("Image provided is a Grayscale image.Quitting!!!!")
        exit(0)
    input_vector = Generator.Color_Moments(image,0) # the input image is then sent to Color Moments generator to get the vector that is then compared with the data avaialble in the database
    print("Testing Color Moments")
    my_cursor = connection.cursor()
    query = "(select image_Id,Feature_Descriptor from Final_Feature_Descriptor);" # Retrieve all the Color Moments of all the images ffom the database
    my_cursor.execute(query) 
    database_result = my_cursor.fetchall() # execute the query and store the database results in the variable
    id_distance = [] #initialize an empty array to store tuple that has image_Id and the distance when compared with the input image
    for DB_item in database_result:
        id_img = DB_item[0]
        FV_string = DB_item[1]
        FV_list_string = FV_string.split(", ")  # the vector in the database are stored as strings, this needs to be splitted into list of strings
        FV_list_float = [float(item) for item in FV_list_string]  #these strings are then need to be converted to float numbers to compute the distance
        distance = distances.euclidean(input_vector,FV_list_float) #here we are using eucledean distance as outr distance function
        id_distance.append((id_img,distance))
    id_distance_sorted = sorted(id_distance, key=lambda x: x[1]) # sort the list based on the distance 
    
    # get all the k images from the sorted list and save the image to the outputs folder
    k_images = []
    for i in range(10):
        image_id = id_distance_sorted[i][0]
        img = dataset[image_id][0]
        k_images.append(img)
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))  # 2 rows and 5 coloumns with each image size as 12,8
    fig.suptitle("Results for Color Moments")
    for i, (img, ax) in enumerate(zip(k_images, axes.flatten())):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1} Score {round(id_distance_sorted[i][1], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/Optimized_project/outputs/Color_Moments/{id}_{timestamp}.jpeg", format="jpeg")
    plt.close()
    print("Finished Color Moments")

#refer to Test_Color_Moments() comments for detailed explanation
def Test_HoG(id,dataset):
    image = dataset[id][0]
    input_vector = Generator.HoG(image,0)
    my_cursor = connection.cursor()
    query = "(select image_Id,HoG_Values from Final_Feature_Descriptor);"
    my_cursor.execute(query)
    database_result = my_cursor.fetchall()
    id_distance = []
    for DB_item in database_result:
        id_img = DB_item[0]
        FV_string = DB_item[1]
        FV_list_string = FV_string.split(", ")
        FV_list_float = [float(item) for item in FV_list_string]
        distance = distances.euclidean(input_vector,FV_list_float)
        id_distance.append((id_img,distance))
    id_distance_sorted = sorted(id_distance, key=lambda x: x[1])
    k_images = []
    for i in range(10):
        image_id = id_distance_sorted[i][0]
        img = dataset[image_id][0]
        k_images.append(img)
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    fig.suptitle("Results for Histogram Of Gradients")
    for i, (img, ax) in enumerate(zip(k_images, axes.flatten())):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1} Score {round(id_distance_sorted[i][1], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/Optimized_project/outputs/Histogram_Of_Gradients/{id}_{timestamp}.jpeg", format="jpeg")
    plt.close()
    print("Testing with histogram of gradients")

#refer to Test_Color_Moments() comments for detailed explanation
def Test_AvgPool(id,dataset):
    image = dataset[id][0]
    input_vector = Generator.AvgPool(image,0)
    my_cursor = connection.cursor()
    query = "(select image_Id,AVG_Pool from Final_Feature_Descriptor);"
    my_cursor.execute(query)
    database_result = my_cursor.fetchall()
    id_distance = []
    for DB_item in database_result:
        id_img = DB_item[0]
        FV_string = DB_item[1]
        FV_list_string = FV_string.split(", ")
        FV_list_float = [float(item) for item in FV_list_string]
        distance = distances.euclidean(input_vector,FV_list_float)
        id_distance.append((id_img,distance))
    id_distance_sorted = sorted(id_distance, key=lambda x: x[1])
    k_images = []
    for i in range(10):
        image_id = id_distance_sorted[i][0]
        img = dataset[image_id][0]
        k_images.append(img)
    k_images.append(image)
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    fig.suptitle("Results for Average Pool Layer")
    for i, (img, ax) in enumerate(zip(k_images, axes.flatten())):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1} Score {round(id_distance_sorted[i][1], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/Optimized_project/outputs/AveragePool/{id}_{timestamp}.jpeg", format="jpeg")
    plt.close()
    print("Testing with avgpool resnet50")

#refer to Test_Color_Moments() comments for detailed explanation
def Test_layer_3(id,dataset):
    image = dataset[id][0]
    input_vector = Generator.layer_3(image,0)
    my_cursor = connection.cursor()
    query = "(select image_Id,layer3 from Final_Feature_Descriptor);"
    my_cursor.execute(query)
    database_result = my_cursor.fetchall()
    id_distance = []
    for DB_item in database_result:
        id_img = DB_item[0]
        FV_string = DB_item[1]
        FV_list_string = FV_string.split(", ")
        FV_list_float = [float(item) for item in FV_list_string]
        distance = distances.euclidean(input_vector,FV_list_float)
        id_distance.append((id_img,distance))
    id_distance_sorted = sorted(id_distance, key=lambda x: x[1])
    k_images = []
    for i in range(10):
        image_id = id_distance_sorted[i][0]
        img = dataset[image_id][0]
        k_images.append(img)
    k_images.append(image)
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    fig.suptitle("Results for Layer 3")
    for i, (img, ax) in enumerate(zip(k_images, axes.flatten())):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1} Score {round(id_distance_sorted[i][1], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/Optimized_project/outputs/Layer3/{id}_{timestamp}.jpeg", format="jpeg")
    plt.close()
    print("Testing layer3 Resnet50")

#refer to Test_Color_Moments() comments for detailed explanation
def Test_FC(id,dataset):
    image = dataset[id][0]
    input_vector = Generator.FC(image,0)
    my_cursor = connection.cursor()
    query = "(select image_Id,FC from Final_Feature_Descriptor);"
    my_cursor.execute(query)
    database_result = my_cursor.fetchall()
    id_distance = []
    for DB_item in database_result:
        id_img = DB_item[0]
        FV_string = DB_item[1]
        FV_list_string = FV_string.split(", ")
        FV_list_float = [float(item) for item in FV_list_string]
        distance = distances.euclidean(input_vector,FV_list_float)
        id_distance.append((id_img,distance))
    id_distance_sorted = sorted(id_distance, key=lambda x: x[1])
    k_images = []
    for i in range(10):
        image_id = id_distance_sorted[i][0]
        img = dataset[image_id][0]
        k_images.append(img)
    k_images.append(image)
    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    fig.suptitle("Results for Resnet FC")
    for i, (img, ax) in enumerate(zip(k_images, axes.flatten())):
        ax.imshow(img)
        ax.set_title(f"Image {i + 1} Score {round(id_distance_sorted[i][1], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/Users/chandu/MS_Computer_Science_ASU/Semester_1/multimedia_web_database/Optimized_project/outputs/FC/{id}_{timestamp}.jpeg", format="jpeg")
    plt.close()
    print("Testing with FC resnet50")