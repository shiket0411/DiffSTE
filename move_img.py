import os
import time
import shutil


def main():
    fake_outpath = "cocotext_deepfake/val/0_fake"
    real_outpath = "cocotext_deepfake/val/1_real"
    os.makedirs(fake_outpath, exist_ok=True)
    os.makedirs(real_outpath, exist_ok=True)

    fake_img_dir_path = 'cocotext_deepfake/train/0_fake'
    real_img_dir_path = 'cocotext_deepfake/train/1_real'
    files = os.listdir(fake_img_dir_path)
    files = files[60003:]

    for file_name in files:
        shutil.move(f'{fake_img_dir_path}/{file_name}', fake_outpath)
        shutil.move(f'{real_img_dir_path}/{file_name}', real_outpath)

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")