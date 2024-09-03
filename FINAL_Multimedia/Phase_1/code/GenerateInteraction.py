import Generator
import torch
def Generate(id,dataset):
    input_image = dataset[id][0]
    model = input(
"""Choose the model to generate your Feature Vector:
    1. RGB Color Moments
    2. Histograms of oriented gradients
    3. ResNet-AvgPool-1024
    4. ResNet-Layer3-1024
    5. ResNet-FC-1000
"""
    )
    if(model not in ["1","2","3","4","5"]):
        print("Invalid Model. Please Select from the List")
        exit(0)
    if(model == "1"):
        Generator.Color_Moments(input_image,1)
        print("Generating Color Moments")
        # print(color_moments)
    elif(model == "2"):
        Generator.HoG(input_image,1)
        print("Generating Histograms of oriented gradients")
        # print(histogram_of_gradients)
    elif(model == "3"):
        average_pool = Generator.AvgPool(input_image,0)
        print("Generating ResNet-AvgPool-1024 (64 x 16)")
        colTesnor = torch.tensor(average_pool).resize(64,16)
        for a in range(64):
            for b in range(16):
                print(str(round(colTesnor[a,b].item(), 4)) + "  ", end='')
            print()
        # print(average_pool)
    elif(model == "4"):
        layer_3_vector = Generator.layer_3(input_image,0)
        print("Generating ResNet-Layer3-1024 (64 x 16)")
        colTesnor = torch.tensor(layer_3_vector).resize(64,16)
        for a in range(64):
            for b in range(16):
                print(str(round(colTesnor[a,b].item(), 4)) + "  ", end='')
            print()
        # print(layer_3_vector)
    elif(model == "5"):
        fc_layer = Generator.FC(input_image,0)
        print("Generating ResNet-FC-1000 (100 x 10)")
        colTesnor = torch.tensor(fc_layer).resize(100,10)
        for a in range(100):
            for b in range(10):
                print(str(round(colTesnor[a,b].item(), 4)) + "  ", end='')
            print()
        # print(fc_layer)