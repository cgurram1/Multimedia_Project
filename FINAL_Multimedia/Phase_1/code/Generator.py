import math
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import torch
import ssl
from torchvision.models import resnet50
from torchvision import transforms

# creating the ssl certificate to load the Resnet50
ssl._create_default_https_context = ssl._create_unverified_context
Resnet50 = resnet50(weights='ResNet50_Weights.DEFAULT') # load the pretrained Resnet50 with default weights
# Defining the Global Variables for Resnet50 intermediate layers to store the intermediate layer results nto these global variables
avgpool_outputs = []
layer3_outputs = []
fc_outputs = []

def Color_Moments(image_large,print_vetor):
    if (image_large.mode not in ('RGB', 'RGBA', 'CMYK', 'YCbCr')):  # check if image is a grascale
        print("The image provided is not a color image")
        dummy_list = []
        for i in range(900):
            dummy_list.append(0)
        return dummy_list # if the image is a grayscale, we return a list of 900 0's
    size = (300, 100)
    image = image_large.resize(size) #resizing the image
    image_matrix = []
    for a in range(10): # specifies the 10 row cells
        image_10_cells = []
        for b in range(10): # specifies the 10 coloumn cells
            cell_matrix = []
            # computing the start and end pixel values for the cell (a,b)
            i_start = 30*a
            j_start = 10*b
            i_end = i_start + 30
            j_end = j_start + 10
            for i in range(i_start,i_end): # each cell has then 30 row pixels
                image_30_pixels = []
                for j in range(j_start,j_end): # each cell has then 10 coloumn pixels
                    RGB = image.getpixel((i,j)) # storing the RGB values for every pixel
                    image_30_pixels.append(RGB)
                cell_matrix.append(image_30_pixels)
            image_10_cells.append(cell_matrix)
        image_matrix.append(image_10_cells) 
    #structure of the image
    # image_matrix[a][b][i][j][k]  ---> gives the (i,j) pixel value from the cell (a,b), K specifies R,G,B
    
    means = []
    sums = []
    SDs = []
    skews = []

    #Mean
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            sums = [0,0,0]
            for i in range(30):
                for j in range(10):
                    for k in range(3):
                        sums[k] = sums[k] + image_matrix[a][b][i][j][k]  # adding all the pixel values
            for k in range(3):
                sums[k] = sums[k]/300 # finding the mean
            image_10_cells.append(sums)
        means.append(image_10_cells)

    #Standard-Deviation
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            sds = [0,0,0]
            for i in range(30):
                for j in range(10):
                    for k in range(3):
                        sds[k] = sds[k] + (image_matrix[a][b][i][j][k] - means[a][b][k])**2
            for k in range(3):
                sds[k] = (sds[k]/300)**(1/2)
            image_10_cells.append(sds)
        SDs.append(image_10_cells)

    #skew
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            skew = [0,0,0]
            for i in range(30):
                for j in range(10):
                    for k in range(3):
                        skew[k] = skew[k] + (image_matrix[a][b][i][j][k] - means[a][b][k])**3
            skew_new = [0,0,0]
            for k in range(3):
                skew_new[k] = skew[k]/300
                skew[k] = math.cbrt(skew_new[k])
            image_10_cells.append(skew)
        skews.append(image_10_cells)

    result_Color_Moments = [means,SDs,skews] # store all the 3 into a single list to get the color mooments vector
    for k in range(3):
        result_Color_Moments = list(chain.from_iterable(result_Color_Moments)) # reshape the array to (900,)
    #Scaling the vector
    original_list = [[x] for x in result_Color_Moments]
    scaler = MinMaxScaler()
    # Fit the scaler on your data to compute the minimum and maximum values
    scaler.fit(original_list)

    # Transform your data to scale it
    scaled_list = scaler.transform(original_list)
    scaled_list = scaled_list.flatten()
    # this if checks if the function is called by the user to display the vector or called by the function to store the results
    if(print_vetor == 1):
        colTesnor = torch.tensor(scaled_list).resize(10,10,9)
        for a in range(10):
            for b in range(10):
                for k in range(9):
                    print(str(round(colTesnor[a,b,k].item(), 2)) + "    ", end='')
                print()
            print()
            print()
    return(scaled_list)

def HoG(image,print_vector):      
    grayscale_image = image.convert("L")
    size = (300, 100)
    resized_image = grayscale_image.resize(size)
    image_matrix = []
    #store the pixel intensities
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            cell_matrix = []
            i_start = 30*a
            j_start = 10*b
            i_end = i_start + 30
            j_end = j_start + 10
            for i in range(i_start,i_end):
                image_30_pixels = []
                for j in range(j_start,j_end):
                    intensity = resized_image.getpixel((i,j))
                    image_30_pixels.append(intensity)
                cell_matrix.append(image_30_pixels)
            image_10_cells.append(cell_matrix)
        image_matrix.append(image_10_cells)
    # calculate Gx Gy for every pixel
    GxGy = []
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            cell_matrix = []
            for i in range(30):
                image_30_pixels = []
                for j in range(10):
                    # These if else cases is a kind of adding padding to the vector, to make sure the list index wont go out of bounds 
                    if((j-1 < 0 or j+1 > 9) and (i-1 >= 0 and i+1 <= 29)):
                        if(j-1 < 0):
                            try:
                                Gx = 0 - image_matrix[a][b][i][j+1]
                            except Exception as e:
                                print("1")
                                print(a,b,i,j)
                        else:
                            try:
                                Gx = image_matrix[a][b][i][j-1] - 0
                            except Exception as e:
                                print("2")
                                print(a,b,i,j)
                        try:
                            Gy = image_matrix[a][b][i-1][j] - image_matrix[a][b][i+1][j]
                        except Exception as e:
                                print("3")
                                print(a,b,i,j)
                    elif((j-1 >= 0 and j+1 <= 9) and (i-1 < 0 or i+1 > 29)):
                        if(i-1 < 0):
                            try:
                                Gy = 0 - image_matrix[a][b][i+1][j]
                            except Exception as e:
                                print("4")
                                print(a,b,i,j)
                        if(i+1 > 29):
                            try:
                                Gy = image_matrix[a][b][i-1][j] - 0
                            except Exception as e:
                                print("5")
                                print(a,b,i,j)
                        try:
                            Gx = image_matrix[a][b][i][j-1] - image_matrix[a][b][i][j+1]
                        except Exception as e:
                                print("6")
                                print(a,b,i,j)
                    elif((j-1 < 0 or j+1 > 9) and (i-1 < 0 or i+1 > 29)):
                        if(j-1 < 0):
                            if(i-1 < 0):
                                try:
                                    Gx = 0 - image_matrix[a][b][i][j+1]
                                except Exception as e:
                                    print("7")
                                    print(a,b,i,j)
                                try:
                                    Gy = 0 - image_matrix[a][b][i+1][j]
                                except Exception as e:
                                    print("8")
                                    print(a,b,i,j)
                            elif(i+1 > 29):
                                try:
                                    Gx = 0 - image_matrix[a][b][i][j+1]
                                except Exception as e:
                                    print("9")
                                    print(a,b,i,j)
                                try:
                                    Gy = image_matrix[a][b][i-1][j] - 0
                                except Exception as e:
                                    print("10")
                                    print(a,b,i,j)
                        elif(j+1 > 9):
                            if(i-1 < 0):
                                try:
                                    Gx = image_matrix[a][b][i][j-1] - 0
                                except Exception as e:
                                    print("11")
                                    print(a,b,i,j)
                                try:
                                    Gy = 0 - image_matrix[a][b][i+1][j]
                                except Exception as e:
                                    print("12")
                                    print(a,b,i,j)
                            elif(i+1 > 29):
                                try:
                                    Gx = image_matrix[a][b][i][j-1] - 0
                                except Exception as e:
                                    print("13")
                                    print(a,b,i,j)
                                try:
                                    Gy = image_matrix[a][b][i-1][j] - 0
                                except Exception as e:
                                    print("14")
                                    print(a,b,i,j)
                    else:
                        try:
                            Gx = image_matrix[a][b][i][j-1] - image_matrix[a][b][i][j+1]
                        except Exception as e:
                            print("15")
                            print(a,b,i,j)
                        try:
                            Gy = image_matrix[a][b][i-1][j] - image_matrix[a][b][i+1][j]
                        except Exception as e:
                            print("16")
                            print(a,b,i,j)
                    image_30_pixels.append([Gx,Gy])
                cell_matrix.append(image_30_pixels)
            image_10_cells.append(cell_matrix)
        GxGy.append(image_10_cells)
    
    # Calulate the Magnitude and angle at every pixel from the Gx Gy
    MagAngle = []
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            cell_matrix = []
            for i in range(30):
                image_30_pixels = []
                for j in range(10):
                    mag = pow((pow((GxGy[a][b][i][j][0]),2) + pow((GxGy[a][b][i][j][1]),2)),1/2)
                    if(GxGy[a][b][i][j][0] == 0):
                        ang = math.degrees(math.atan((GxGy[a][b][i][j][1])/(GxGy[a][b][i][j][0] + 1e-10)))
                    else:
                        ang = math.degrees(math.atan((GxGy[a][b][i][j][1])/(GxGy[a][b][i][j][0])))
                    ang = (ang + 360) % 360
                    image_30_pixels.append([mag,ang])
                cell_matrix.append(image_30_pixels)
            image_10_cells.append(cell_matrix)
        MagAngle.append(image_10_cells)

    #9-point histogram
    # number_of_bins = 9
    # step_size = 360/40
    HoG = []
    for a in range(10):
        image_10_cells = []
        for b in range(10):
            bins_index_9 = [0,0,0,0,0,0,0,0,0]
            for i in range(30):
                for j in range(10):
                    Angle = MagAngle[a][b][i][j][1]
                    mag = MagAngle[a][b][i][j][0]
                    bin_index = int(Angle / 40.0) % 9
                    bins_index_9[bin_index] = bins_index_9[bin_index] + mag  # go to the index and add the magnitude corresponding to that bin
            image_10_cells.append(bins_index_9)
        HoG.append(image_10_cells)
    result_HoG = list(chain.from_iterable(HoG))
    result_HoG = list(chain.from_iterable(result_HoG)) # reshape the vector to (900,)
    if(print_vector == 1):
        colTesnor = torch.tensor(result_HoG).resize(10,10,9)
        for a in range(10):
            for b in range(10):
                for k in range(9):
                    print(str(round(colTesnor[a,b,k].item(), 2)) + "    ", end='')
                print()
            print()
            print()
    return(result_HoG)

def AvgPool(image,curr_image_no):
    Resnet50.eval()
    transformGrayScale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if (image.mode not in ('RGB', 'RGBA', 'CMYK', 'YCbCr')):
        resized_image = transformGrayScale(image).unsqueeze(0)
    else:
        resized_image = transform(image).unsqueeze(0)
    Resnet50.avgpool.register_forward_hook(AvgPool_hook_fn)
    with torch.no_grad():
        output = Resnet50(resized_image)
    avgpool_output = avgpool_outputs[curr_image_no]
    reshaped_vector = avgpool_output.view(-1, 2)
    reduced_vector = torch.mean(reshaped_vector, dim=1)
    vector_json = reduced_vector.tolist()
    return(vector_json)  
def AvgPool_hook_fn(module, input, output):
    avgpool_outputs.append(output)

def layer_3(image,curr_image_no):
    Resnet50.eval()
    transformGrayScale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if (image.mode not in ('RGB', 'RGBA', 'CMYK', 'YCbCr')):
        resized_image = transformGrayScale(image).unsqueeze(0)
    else:
        resized_image = transform(image).unsqueeze(0)

    Resnet50.layer3.register_forward_hook(layer3_hook_fn)
    with torch.no_grad():
        output = Resnet50(resized_image)
    layer3_output = layer3_outputs[curr_image_no]
    layer3_output = layer3_output.squeeze(0)
    reduced_slices = torch.mean(layer3_output, dim=(1, 2))
    reduced_vector = reduced_slices.view(-1)
    vector_json = reduced_vector.tolist()
    return(vector_json)
def layer3_hook_fn(module, input, output):
    layer3_outputs.append(output)

def FC(image,curr_image_no):
    Resnet50.eval()
    transformGrayScale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if (image.mode not in ('RGB', 'RGBA', 'CMYK', 'YCbCr')):
        resized_image = transformGrayScale(image).unsqueeze(0)
    else:
        resized_image = transform(image).unsqueeze(0)

    Resnet50.fc.register_forward_hook(fc_hook_fn)
    with torch.no_grad():
        output = Resnet50(resized_image)
    fc_output = fc_outputs[curr_image_no]
    fc_output = fc_output.squeeze(0)
    vector_json = fc_output.tolist()
    return(vector_json)
def fc_hook_fn(module, input, output):
    fc_outputs.append(output)