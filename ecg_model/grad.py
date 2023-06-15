from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def ecg_grad(model, input_img_pred, image):
    global gradients
    global activations
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        global gradients # refers to the variable in the global scope

        gradients = grad_output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])

        # We need the 0 index because the tensor containing the gradients comes
        # inside a one element tuple.
        # defines two global scope variables to store our gradients and activations

    def forward_hook(module, args, output):
        global activations # refers to the variable in the global scope
        activations = output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])

    # Get the final layer of the ResNet model
    final_layer = model.layer4[-1]
    #hooks
    backward_hook = final_layer.register_full_backward_hook(backward_hook)
    forward_hook = final_layer.register_forward_hook(forward_hook)

    output = model(input_img_pred)
    output[0][0].backward()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    # weight the channels by corresponding gradients
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = F.relu(heatmap)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Create a figure and plot the first image
    fig, ax = plt.subplots()
    ax.axis('off') # removes the axis markers
    # First plot the original image
    image_1000 = image.resize((1000, 1000))
    ax.imshow(image_1000)

    # Resize the heatmap to the same size as the input image and defines
    # a resample algorithm for increasing image resolution
    # we need heatmap.detach() because it can't be converted to numpy array while
    # requiring gradients
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((1000,1000), resample=PIL.Image.BICUBIC)
    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Plot the heatmap on the same axes,
    # but with alpha < 1 (this defines the transparency of the heatmap)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest')

    imagen = plt.savefig('ecg_model/api/grad_cam.jpg')

    # Remove the hooks when done
    backward_hook.remove()
    forward_hook.remove()

    return imagen
