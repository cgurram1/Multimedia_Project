import ssl
from torchvision.models import resnet50
from torchvision import transforms
ssl._create_default_https_context = ssl._create_unverified_context
import torch


# This is completely used only for storing to the database.
avgpool_outputs = []
layer3_outputs = []
fc_outputs = []
def allLayers(image,curr_image_no):
    Resnet50 = resnet50(weights='ResNet50_Weights.DEFAULT')
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
    Resnet50.layer3.register_forward_hook(layer3_hook_fn)
    Resnet50.fc.register_forward_hook(fc_hook_fn)
    with torch.no_grad():
        Resnet50(resized_image)
    
    avgpool_output = avgpool_outputs[curr_image_no]
    avgpool_reshaped_vector = avgpool_output.view(-1, 2)
    avgpool_reduced_vector = torch.mean(avgpool_reshaped_vector, dim=1)
    avgpool_vector_json = avgpool_reduced_vector.tolist()


    layer3_output = layer3_outputs[curr_image_no]
    layer3_output = layer3_output.squeeze(0)
    layer3_reduced_slices = torch.mean(layer3_output, dim=(1, 2))
    layer3_reduced_vector = layer3_reduced_slices.view(-1)
    layer3_vector_json = layer3_reduced_vector.tolist()

    fc_output = fc_outputs[curr_image_no]
    fc_output = fc_output.squeeze(0)
    fc_vector_json = fc_output.tolist()

    return([avgpool_vector_json,layer3_vector_json,fc_vector_json])
def AvgPool_hook_fn(module, input, output):
    avgpool_outputs.append(output)
def layer3_hook_fn(module, input, output):
    layer3_outputs.append(output)
def fc_hook_fn(module, input, output):
    fc_outputs.append(output)