import cv2
import os

def mp4_to_png_1(input_mp4, output_folder='./data/BDD100k_s/BDD100k_s'):
    cv2.setRNGSeed(42) 

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_mp4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'mp4v'))
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)  
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{frame_count:03d}.png")
        cv2.imwrite(output_path, frame)
        frame_count += 1
    
    cap.release()
   

def mp4_to_png_2(input_mp4, output_folder='./output/tmp/LR/LR'):
    cv2.setRNGSeed(42) 

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_mp4)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'mp4v'))
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)  
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_folder, f"frame_{frame_count:03d}.png")
        cv2.imwrite(output_path, frame)
        frame_count += 1
    
    cap.release()
    

def png_to_mp4_1(output_mp4, fps,input_folder='./output/LR/LR'):
    cv2.setRNGSeed(42)  

    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    first_image = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Finish: {output_mp4}")

def png_to_mp4_2(output_mp4, fps,input_folder='./output/Restore'):
    cv2.setRNGSeed(42)  

    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])
    first_image = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Finish: {output_mp4}")

