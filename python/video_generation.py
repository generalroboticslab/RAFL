import os
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

def generate_video_directory(fig_path, path_nums, flag="", delete_after=False):

    vid_folder = os.path.join(fig_path, flag+'videos') if not delete_after else fig_path
    if not delete_after:
        Path(vid_folder).mkdir(parents=True, exist_ok=True)

    print("Creating Videos")
    for i in tqdm(path_nums):

        generate_video(fig_path, str(i), os.path.join(vid_folder, f'{i}.mp4'), delete_after=delete_after)


def generate_video(fig_path, path_num, dst_path, fps=20, delete_after=False):

    fig_path = os.path.join(fig_path, path_num)

    images = [img for img in os.listdir(fig_path)
                if img.endswith(".png") or img.endswith(".jpg")]
    
    images.sort(key = lambda x: int(x[:-4]))

    frame = cv2.imread(os.path.join(fig_path, images[0]))

    height, width, _ = frame.shape  

    video = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) 

    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(fig_path, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

    if delete_after:
        shutil.rmtree(fig_path)