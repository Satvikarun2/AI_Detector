# preprocess by converting images into fingerprints (Spectral, ELA, PRNU) and save to disk
import numpy as np
import cv2
import PIL.Image
import h5py
from utils import azi_diff 
from tqdm import tqdm
import os
import random
import pickle
import logging
import joblib

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
    for directory in tqdm(class1_dirs, desc="Loading Real images"):
        class1_files.extend(get_image_files(directory))
    class2_files = []
    for directory in tqdm(class2_dirs, desc="Loading Fake images"):
        class2_files.extend(get_image_files(directory))
    
    # Ensure equal representation (Balance dataset)
    min_length = min(len(class1_files), len(class2_files))
    random.shuffle(class1_files)
    random.shuffle(class2_files)
    class1_files = class1_files[:min_length]
    class2_files = class2_files[:min_length]
    print(f"Number of files: Real = {len(class1_files)}, Fake = {len(class2_files)}")
    return class1_files, class2_files

def process_and_save_h5(file_label_pairs, patch_num, N, save_interval, joblib_batch_size, output_dir, start_by=0):
    def process_file(file_label):
        path, label = file_label
        try:
            # Open image as PIL once and pass to processing
            img = PIL.Image.open(path).convert('RGB')
            # Updated azi_diff returns: spectral, ela, and noise
            result = azi_diff(img, patch_num, N) 
            return result, label
        except Exception as e:
            logging.error(f"Error processing file {path}: {str(e)}")
            return None, None

    num_files = len(file_label_pairs)
    num_saves = (num_files - start_by + save_interval - 1) // save_interval
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    

    with tqdm(total=num_files - start_by, desc="Total Progress", unit="image") as pbar:
        for save_index in range(num_saves):
            save_start = start_by + save_index * save_interval
            save_end = min(save_start + save_interval, num_files)
            batch_pairs = file_label_pairs[save_start:save_end]
            
            all_rich, all_poor, all_ela, all_noise, all_labels = [], [], [], [], []

            for batch_start in range(0, len(batch_pairs), joblib_batch_size):
                batch_end = min(batch_start + joblib_batch_size, len(batch_pairs))
                small_batch_pairs = batch_pairs[batch_start:batch_end]
                
                processed_data = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(process_file)(file_label) for file_label in small_batch_pairs
                )

                for data, label in processed_data:
                    if data is not None:
                        all_rich.append(data['total_emb'][0])
                        all_poor.append(data['total_emb'][1])
                        all_ela.append(data['ela'])
                        all_noise.append(data['noise'])
                        all_labels.append(label)
                
                pbar.update(len(small_batch_pairs))

            output_filename = os.path.join(output_dir, f"processed_data_{save_end}.h5")
            logging.info(f"Saving {output_filename}")
            
            with h5py.File(output_filename, 'w') as h5file:
                h5file.create_dataset('rich', data=np.array(all_rich))
                h5file.create_dataset('poor', data=np.array(all_poor))
                h5file.create_dataset('ela', data=np.array(all_ela))
                h5file.create_dataset('noise', data=np.array(all_noise))
                h5file.create_dataset('labels', data=np.array(all_labels))
                            
            logging.info(f"Successfully saved {output_filename}")
            del all_rich, all_poor, all_ela, all_noise, all_labels

# --- Configuration for your AI_Detector Manual Uploads ---
load = False
class1_dirs = ["./data/real/"] # Manually put Real images here (Label 0)
class2_dirs = ["./data/ai/"]   # Manually put AI images here (Label 1)

output_dir = "./h5_storage"
file_paths_pickle_save_dir = './aigc_multimodal_paths.pkl'

patch_num = 128
N = 256
save_interval = 2000
joblib_batch_size = 400
start_by = 0

# --- Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if load:
        if os.path.exists(file_paths_pickle_save_dir):
            with open(file_paths_pickle_save_dir, 'rb') as file:
                file_label_pairs = pickle.load(file)
            print(f"Loaded {len(file_label_pairs)} file pairs.")
        else:
            print("Pickle file not found. Set load=False to scan directories.")
    else:
        class1_files, class2_files = load_image_files(class1_dirs, class2_dirs)
        file_label_pairs = list(zip(class1_files, [0] * len(class1_files))) + \
                           list(zip(class2_files, [1] * len(class2_files)))
        random.shuffle(file_label_pairs)
        with open(file_paths_pickle_save_dir, 'wb') as file:
            pickle.dump(file_label_pairs, file)
        print(f"Saved path map to {file_paths_pickle_save_dir}")

    process_and_save_h5(file_label_pairs, patch_num, N, save_interval, joblib_batch_size, output_dir, start_by)