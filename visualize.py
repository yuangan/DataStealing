from PIL import Image
import numpy as np

def visualize_img(img, path):
    # Convert the value range from [-1, 1] to [0, 1]
    img = (img + 1) / 2
    # Convert the value range from [0, 1] to [0, 255]
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    # Change the shape from [1, 3, 224, 224] to [224, 224, 3]
    img = img.transpose((2, 3, 1, 0)).squeeze()

    # Then, we need to convert it to a PIL Image object
    img = Image.fromarray(img)

    # Finally, we can save it as an image file
    img.save(path)

def visualize_depth_as_png(data, filename):
    """
    All depths are saved in np.uint16
    """
    data = -data
    data = (data - data.min()) / ((data.max() - data.min()))
    data = data.clamp(0.0, 1.0).repeat(1, 3, 1, 1).squeeze(0).permute(1,2,0)
    data_np = data.detach().cpu().float().numpy()*255
    data_np = data_np.astype(np.uint8)
    data_pil = Image.fromarray(data_np)
    data_pil.save(filename) 

def visualize_mask(img, path):
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = np.repeat(img.transpose((1, 2, 0)), 3, axis=2)
    # Then, we need to convert it to a PIL Image object
    img = Image.fromarray(img)
    # Finally, we can save it as an image file
    img.save(path)

