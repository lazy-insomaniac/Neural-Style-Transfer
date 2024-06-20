import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define image size based on device
size = 512 if torch.cuda.is_available() else 128
# Normalization and denormalization transformations
model_normalization_mean = [0.485, 0.456, 0.406]
model_normalization_std = [0.229, 0.224, 0.225]

loader = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=model_normalization_mean, std=model_normalization_std)
])

unloader = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(model_normalization_mean, model_normalization_std)], std=[1/s for s in model_normalization_std]),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  
    transforms.ToPILImage()
])

def img_loader(img_name):
    image = Image.open(img_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def unloader_view(tensor):
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    return img

def content_loss(input, target):
    target = target.detach()
    return F.mse_loss(input, target)

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def style_loss(input, target):
    i_gram = gram_matrix(input)
    t_gram = gram_matrix(target).detach()
    return F.mse_loss(i_gram, t_gram)
# Load VGG19 model
weights = models.VGG19_Weights.IMAGENET1K_V1
model = models.vgg19(weights=weights).features.to(device).eval()

def set_relu_inplace_false(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

# Apply the function to the model (this is done because the layers cause problem to the content loss function)
set_relu_inplace_false(model)

def forward(x, layer_indices, model):
    output = []
    for i, layer in enumerate(model):
        x = layer(x)
        if i in layer_indices:
            output.append(x)
    return output

def display_side_by_side(content_img, style_img, gen_img=None, stage='before'):
    content_img = np.array(unloader_view(content_img))
    style_img = np.array(unloader_view(style_img))
    
    if gen_img is not None:
        gen_img = np.array(unloader_view(gen_img))
    
    fig, ax = plt.subplots(1, 5 if gen_img is not None else 4, figsize=(15, 5))
    titles = ["CONTENT IMAGE", "", "STYLE IMAGE", "", "RESULT"] if stage == 'after' else ["CONTENT IMAGE", "", "STYLE IMAGE", "", ""]
    for i in range(4):
        ax[i].axis('off')
        if i % 2 == 0:
            ax[i].set_title(titles[i])
    ax[0].imshow(content_img)
    ax[0].axis('off')
    ax[1].text(0.5, 0.5, '+', fontsize=30, ha='center')
    ax[1].axis('off')
    ax[2].imshow(style_img)
    ax[2].axis('off')
    ax[3].text(0.5, 0.5, '=' if stage == 'after' else '?', fontsize=30, ha='center')
    ax[3].axis('off')
    
    if gen_img is not None:
        ax[4].imshow(gen_img)
        ax[4].axis('off')
    
    plt.show()

def run_style_transfer(content_img_path, style_img_path, epochs=1000, style_weight=1e6, content_weight=1, lr=0.1):
    con_img = img_loader(content_img_path)
    sty_img = img_loader(style_img_path)
    gen_img = con_img.clone().requires_grad_(True)
    
    print("Initial Images")
    
    display_side_by_side(con_img, sty_img, stage='before')
    
    print("Neural Style Transfer Ongoing....")
    optimizer = torch.optim.Adam([gen_img], lr=lr)

    content_layers = [22]
    style_layers = [1, 6, 11, 20, 29]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        style_features = forward(sty_img, style_layers, model)
        content_features = forward(con_img, content_layers, model)
        gen_style_features = forward(gen_img, style_layers, model)
        gen_content_features = forward(gen_img, content_layers, model)
        
        style_losses = [style_loss(gs, s) for gs, s in zip(gen_style_features, style_features)]
        content_losses = [content_loss(gc, c) for gc, c in zip(gen_content_features, content_features)]
        
        total_loss = style_weight * sum(style_losses) + content_weight * sum(content_losses)
        total_loss.backward()
        
        optimizer.step()
    
    print("Final Images")
    display_side_by_side(con_img, sty_img, gen_img, stage='after')
    
    return gen_img

# Example usage
content_img_path = '/kaggle/input/neural-style-transfer/content3.jpeg'  # Path to the content image
style_img_path = '/kaggle/input/neural-style-transfer/style4.jpeg'    # Path to the style image

result_img = run_style_transfer(content_img_path, style_img_path, epochs=1000, style_weight=1e6, content_weight=1, lr=0.01)

# Save the generated image
content_img_name = content_img_path.split('/')[-1].split('.')[0]
style_img_name = style_img_path.split('/')[-1].split('.')[0]
result_img = unloader(result_img.clone().cpu().squeeze(0))
print("RESULTANT IMAGE")
plt.imshow(result_img)
print("SAVING AS ",content_img_name,"+",style_img_name,".png")
result_img.save(f'{content_img_name}+{style_img_name}.png')