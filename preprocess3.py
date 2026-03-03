import numpy as np
import PIL.Image
import h5py
from utils import azi_diff 
from tqdm import tqdm
import os
import random
import logging
import joblib

# --- Helper Functions ---
def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files

def load_image_files(class1_dirs, class2_dirs):
    class1_files = []
    for directory in class1_dirs:
        class1_files.extend(get_image_files(directory))
    class2_files = []
    for directory in class2_dirs:
        class2_files.extend(get_image_files(directory))
    
    # SHUFFLE ONLY: Removed min_length truncation to keep all 16 real and 7 fake
    random.shuffle(class1_files)
    random.shuffle(class2_files)
    
    print(f"✅ New Unseen Files Loaded: Real = {len(class1_files)}, Fake = {len(class2_files)}")
    return class1_files, class2_files

def process_and_save_h5(file_label_pairs, patch_num, N, output_dir):
    def process_file(file_label):
        path, label = file_label
        try:
            img = PIL.Image.open(path).convert('RGB')
            result = azi_diff(img, patch_num, N) 
            return result, label
        except Exception:
            return None, None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_files = len(file_label_pairs)
    with tqdm(total=num_files, desc="Processing New Unseen Data", unit="image") as pbar:
        processed_data = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_file)(file_label) for file_label in file_label_pairs
        )

        all_rich, all_poor, all_ela, all_noise, all_labels = [], [], [], [], []
        for data, label in processed_data:
            if data is not None:
                all_rich.append(data['total_emb'][0])
                all_poor.append(data['total_emb'][1])
                all_ela.append(data['ela'])
                all_noise.append(data['noise'])
                all_labels.append(label)
        
        output_filename = os.path.join(output_dir, "unseen_test_data_2.h5")
        with h5py.File(output_filename, 'w') as h5file:
            h5file.create_dataset('rich', data=np.array(all_rich))
            h5file.create_dataset('poor', data=np.array(all_poor))
            h5file.create_dataset('ela', data=np.array(all_ela))
            h5file.create_dataset('noise', data=np.array(all_noise))
            h5file.create_dataset('labels', data=np.array(all_labels))
        
        print(f"✨ Successfully saved processed unseen data to {output_filename}")

# --- CONFIGURATION ---
class1_dirs = ["./data/improv test real images/"] 
class2_dirs = ["./data/improv test fake images/"]   
output_dir = "./h5_unseen2"

patch_num = 128
N = 256

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    class1_files, class2_files = load_image_files(class1_dirs, class2_dirs)
    file_label_pairs = list(zip(class1_files, [0] * len(class1_files))) + \
                       list(zip(class2_files, [1] * len(class2_files)))
    random.shuffle(file_label_pairs)
    
    process_and_save_h5(file_label_pairs, patch_num, N, output_dir)