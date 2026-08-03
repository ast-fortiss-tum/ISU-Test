from PIL import Image, ImageEnhance
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
base_result_dir = PROJECT_ROOT / "data_val"
image_folder = base_result_dir / "images"

output_folder_reproduced = base_result_dir / "images_reproduced"
#output_folder_sim2real = base_result_dir / "images_sim2real"
output_folder_sim = base_result_dir / "images_sim"

NOOP = True

def add_yellow_cast(image):
    arr = np.array(image).astype(float)

    # Increase red + green channels → warmer / yellowish
    arr[..., 0] *= 1.08   # red ↑
    arr[..., 1] *= 1.05   # green ↑

    # Clip values to valid range
    arr = np.clip(arr, 0, 255).astype('uint8')
    warmed = Image.fromarray(arr)

    # Optionally increase saturation slightly (looks more natural)
    enhancer = ImageEnhance.Color(warmed)
    warmed = enhancer.enhance(1.05)  # 1.0 = none, >1 = more saturation

    return warmed

def remove_yellow_cast(image):
    # Convert to numpy array
    arr = np.array(image).astype(float)

    # Slightly boost blue channel, reduce red a bit
    arr[..., 2] *= 1.10   # blue ↑
    arr[..., 0] *= 0.95   # red ↓

    # Clip to valid range
    arr = np.clip(arr, 0, 255).astype('uint8')
    corrected = Image.fromarray(arr)

    # Optional: reduce overall warm tint further via saturation control
    enhancer = ImageEnhance.Color(corrected)
    corrected = enhancer.enhance(0.9)  # lower saturation slightly

    return corrected

def noop(image):
    return image


#brightness_factor = 1.5

#output_folder_reproduced.mkdir(parents=True, exist_ok=True)
#output_folder_sim2real.mkdir(parents=True, exist_ok=True)
# output_folder_sim.mkdir(parents=True, exist_ok=True)

# for image_name in os.listdir(image_folder):
#     if image_name.endswith("_reproduced.png"):
#         src_path = image_folder / image_name
#         dst_path = output_folder_reproduced / image_name

#         img = Image.open(src_path)

#         if NOOP:
#             bright_img = noop(img)
#         else:
#             enhancer = ImageEnhance.Brightness(img)
#             bright_img = enhancer.enhance(brightness_factor)

#         bright_img.save(dst_path)

    # if image_name.endswith("_sim2real.png"):
    #     src_path = image_folder / image_name
    #     dst_path = output_folder_sim2real / image_name

    #     if NOOP:
    #         corrected = noop(Image.open(src_path))
    #     else:
    #         img = Image.open(src_path)
    #         corrected = remove_yellow_cast(img)

    #     corrected.save(dst_path)

    # if image_name.endswith("_sim.png"):
    #     src_path = image_folder / image_name
    #     dst_path = output_folder_sim / image_name

    #     if NOOP:
    #         warmed = noop(Image.open(src_path))
    #     else:
    #         img = Image.open(src_path)
    #         warmed = add_yellow_cast(img)

    #     warmed.save(dst_path)

# plot images with same base name side by side for comparison from different domain folders
import matplotlib.pyplot as plt

# for image_name in os.listdir(output_folder_reproduced):
#     if image_name.endswith("_reproduced.png"):
#         base_name = image_name.replace("_reproduced.png", "")

#         img_reproduced = Image.open(output_folder_reproduced / image_name)
#         img_sim2real = Image.open(output_folder_sim2real / f"{base_name}_sim2real.png")
#         img_sim = Image.open(output_folder_sim / f"{base_name}_sim.png")

#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#         axs[0].imshow(img_sim)
#         axs[0].set_title("Sim")
#         axs[0].axis('off')

#         axs[1].imshow(img_sim2real)
#         axs[1].set_title("Sim2Real")
#         axs[1].axis('off')

#         axs[2].imshow(img_reproduced)
#         axs[2].set_title("Reproduced")
#         axs[2].axis('off')

#         plt.suptitle(f"Image Comparison: {base_name}")
#         plt.show()

#print all 20 images pairs together in one single plot for comparison, instead of showing one by one
#resize to same height first, preserve original aspect ratio
image_names = [img_name for img_name in os.listdir(output_folder_reproduced) if img_name.endswith("_reproduced.png")]
num_images = len(image_names)
fig, axs = plt.subplots(num_images, 2, figsize=(14, num_images * 5), constrained_layout=True)

def resize_by_height(img, target_h):
    w, h = img.size
    new_w = int(w * (target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)

for i, image_name in enumerate(image_names):
    base_name = image_name.replace("_reproduced.png", "")

    img_reproduced = Image.open(output_folder_reproduced / image_name)
    #img_sim2real = Image.open(output_folder_sim2real / f"{base_name}_sim2real.png")
    img_sim = Image.open(output_folder_sim / f"{base_name}_sim.png")
    target_height = 240


    img_sim = resize_by_height(img_sim, target_height)
    # img_sim2real = resize_by_height(img_sim2real, target_height)
    img_reproduced = resize_by_height(img_reproduced, target_height)

    axs[i, 0].imshow(img_sim)
    axs[i, 0].axis('off')

    # axs[i, 1].imshow(img_sim2real)
    # axs[i, 1].axis('off')

    axs[i, 1].imshow(img_reproduced)
    axs[i, 1].axis('off')

# Add column titles only once
axs[0, 0].set_title("Sim", fontsize=14, pad=10)
#axs[0, 1].set_title("Sim2Real", fontsize=14, pad=10)
axs[0, 1].set_title("Reproduced", fontsize=14, pad=10)

fig.suptitle("Image Comparison Across Domains", fontsize=18)
output_path = PROJECT_ROOT / "data_val" / "image_color_correction_comparison.png"
plt.savefig(output_path, dpi=200)
plt.show()

print(f"Saved to: {output_path}")