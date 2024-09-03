import Testing_images
def Test(id,dataset):
    model = input(
"""Choose the Vector Space with which you want to test the image:
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
        Testing_images.Test_Color_Moments(id,dataset) #Choose this if you want to test with Color Moments
    elif(model == "2"):
        Testing_images.Test_HoG(id,dataset) # choose this if you want to test with Histogram Of Gradients
    elif(model == "3"):
        Testing_images.Test_AvgPool(id,dataset) # choose this if you want to test with the resnet50's average pool layer
    elif(model == "4"):
        Testing_images.Test_layer_3(id,dataset) # choose this if you want to test with the resnet50's layer 3
    elif(model == "5"):
        Testing_images.Test_FC(id,dataset) # choose this if you want to test with the resnet50's FC layer