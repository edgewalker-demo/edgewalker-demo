import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
from skimage import io
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import torch
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def color_quantization():
# Due to hte color sampling used in kmeans (image_array_sample = shuffle(image_array, random_state=0)[:1000]),
# even using more than 24 bits cannot get original picture
    n_colors = 16
    import os

  
    # input and output patch
    folder_path = './output/tmp/after_ae_encoder/after_ae_encoder'
    output_folder = './output/tmp/aftercc/aftercc'
    os.makedirs(output_folder, exist_ok=True)
    #get file
    file_list = os.listdir(folder_path)

    image_extensions = ['.png']
    image_files = [f for f in file_list if any(f.endswith(ext) for ext in image_extensions)]

    # read file
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        imgs = io.imread(image_path)
        
        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        imgs = np.array(imgs, dtype=np.float64) / 255
        
        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(imgs.shape)
        # print(w)
        # print(d)
        #assert d == 3 #4
        image_array = np.reshape(imgs, (w * h, d))
        
        # print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        # print("done in %0.3fs." % (time() - t0))
        
        # Get labels for all points
        # print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        # print("done in %0.3fs." % (time() - t0))
        
        
        codebook_random = shuffle(image_array, random_state=0)[:n_colors]
        # print("Predicting color indices on the full image (random)")
        t0 = time()
        labels_random = pairwise_distances_argmin(codebook_random,
                                                image_array,
                                                axis=0)
        # print("done in %0.3fs." % (time() - t0))
        
        
        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            d = codebook.shape[1]
            image = np.zeros((w,h,d))
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = codebook[labels[label_idx]]
                    label_idx += 1
            return image
        
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.title('Original frame')
        # plt.imshow(imgs)
        
        plt.figure(2)
        plt.clf()
        plt.axis('off')
        # plt.title('Quantized image (64 colors, K-Means)')
        plt.title('Quantized frame ({} colors)'.format(n_colors))
        # plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
        io.imsave(output_image_path,
                np.uint8(recreate_image(kmeans.cluster_centers_, labels, w, h)*255))
        # plt.show()

    root = "./output/tmp/aftercc"
    LR_size = [90, 160]  
    output_path = './output/LR/LR'
    os.makedirs(output_path, exist_ok=True)

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    dataloader = DataLoader(
        torchvision.datasets.ImageFolder(root, transform=transforms_),
        batch_size=batch_size,  
        shuffle=False, 
        num_workers=4,  
        pin_memory=True,
        prefetch_factor=4
    )

    resize_transform = transforms.Resize(LR_size)  
    image_index = 100  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)  
        images_resized = resize_transform(images)  

        
        for i in range(images_resized.shape[0]):  
            filename = f"{image_index}.png"  
            file_path = os.path.join(output_path, filename)
            torchvision.utils.save_image(images_resized[i], file_path, normalize=True)

            image_index += 1  
    return
