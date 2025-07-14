import os
import re
import random
import numpy as np
from PIL import Image, ImageDraw

def random_crop(img_path, folder_path = "./", idx=None):
    img = Image.open(img_path)
    W, H = img.size
    crop_ratio = random.uniform(1/4, 1/2)
    crop_width = int(W * crop_ratio)
    crop_height = int(H * crop_ratio)
    start_x = random.randint(0, W - crop_width)
    start_y = random.randint(0, H - crop_height)
    cropped_image = img.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))
    cropped_image.save(folder_path + 'random_crop_' + str(idx) + '.jpg')

    return folder_path + 'random_crop_' + str(idx) + '.jpg'

def crop_image(img_path, folder_path = "./", idx=None):
    img = Image.open(img_path)
    new_width = img.width // 2
    new_height = img.height // 2

    center_x, center_y = img.width // 2, img.height // 2

    left = center_x - new_width // 2
    top = center_y - new_height // 2
    right = center_x + new_width // 2
    bottom = center_y + new_height // 2

    center_img = img.crop((left, top, right, bottom))
    center_img.save(folder_path + 'center_crop.jpg')

    return folder_path + 'center_crop.jpg'

def remove_areas(img, discard_boxes):
    img_array = np.array(img)

    for discard_box in discard_boxes:
        left, top, right, bottom = discard_box

        img_array[top:bottom, left:right] = 0
    new_img = Image.fromarray(img_array)

    return new_img

def auto_mask(model, img_path, folder_path = "./", idx=None):
    img = Image.open(img_path)

    seg = "Return the coordinates of the text box in the image with the following format: {'obj1': [left, top, right, bottom], 'obj2': ...}."
    notice = f"Notice that the original image shape is {[0, 0, img.size[0], img.size[1]]}. Your coordinates must not exceed the original size.\n"
    model.ask(notice + seg + "Direct give your answer:", img_path)

    text = model.memory_lst[-1]['content']
    pattern = r'\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, text)
    coords = [list(match) for match in matches]
    image_list = []
    for i in range(len(coords)):
        coords[i] = [
            max(0, int(coords[i][0])),
            max(0, int(coords[i][1])),
            min(img.size[0], int(coords[i][2])),
            min(img.size[1], int(coords[i][3]))
        ]

    auto_img = remove_areas(img, coords)
    auto_img.save(folder_path + 'auto_crop_' + str(idx) + '.jpg')
    image_list.append(folder_path + 'auto_crop_' + str(idx) +'.jpg')

    return image_list

def auto_crop(model, img_path, folder_path = "./", idx=None):
    if isinstance(img_path, Image.Image):
        img = img_path
        case_property = 'default_case'
        end = idx
    elif isinstance(img_path, str):
        img = Image.open(img_path)
        case_property = img_path.split('/')[-2]
        end = img_path.split('/')[-1]
    else:
        raise ValueError("Input must be a PIL Image object or a valid image file path.")

    seg = "Segment the main objects and return their coordinates in the following format: {'obj1': [left, top, right, bottom], 'obj2':...}."
    notice = f"Notice that the original image shape is {[0, 0, img.size[0], img.size[1]]}. Your coordinates must not exceed the original size.\n"
    model.ask(notice + seg + "Direct give your answer:", img_path)

    text = model.memory_lst[-1]['content']
    pattern = r'\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, text)
    coords = [list(match) for match in matches]
    image_list = []
    for i in range(len(coords)):
        coords[i] = [
            max(0, int(coords[i][0])),
            max(0, int(coords[i][1])),
            min(img.size[0], int(coords[i][2])),
            min(img.size[1], int(coords[i][3]))
        ]
        auto_img = img.crop((int(coords[i][0]), int(coords[i][1]), int(coords[i][2]), int(coords[i][3])))

        auto_img.save(folder_path + 'center_crop.jpg')
        image_list.append(folder_path + 'center_crop.jpg')
    return image_list

def get_random_smooth(dataset, image_path, num_images=10, perturbation_percentage=0.2, folder_path="/root/Multi-agent Debate/smooth/"):
    image_name = image_path.split('.')[-2].split('/')[-1]
    img = Image.open(image_path)
    width, height = img.size
    if (width * height) / (1024*1024) > 5:
        width /= 10
        height /= 10
        width = int(width)
        height = int(height)
        img = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    if not os.path.exists(folder_path+dataset):
        os.makedirs(folder_path+dataset)
    for i in range(num_images):
        mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        transparency = 255
        color = (0, 0, 0)
        total_pixels = width * height
        num_masked_pixels = int(total_pixels * perturbation_percentage)
        masked_positions = random.sample(range(total_pixels), num_masked_pixels)
        for pos in masked_positions:
            x = pos % width
            y = pos // width
            mask.putpixel((x, y), (color[0], color[1], color[2], transparency))
        masked_image = Image.alpha_composite(img.convert('RGBA'), mask)
        masked_image.save(f"./{image_name}_{i}.png")

    return f"./{image_name}"