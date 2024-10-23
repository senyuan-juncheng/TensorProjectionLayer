import os
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import numpy as np

# フォルダをクリア
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# 同じ変換を画像とマスクに適用するジェネレータ
def create_augmented_images(train_df, image_size, save_to_dir_images, save_to_dir_masks, num_aug=5):
    # フォルダを空にする
    clear_directory(save_to_dir_images)
    clear_directory(save_to_dir_masks)
    
    # ImageDataGeneratorでデータ拡張
    data_gen_args = dict(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.05,
        brightness_range=[0.8, 1.2],  # 明るさの変化
        channel_shift_range=20,         # 色合いのシフト
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
    )
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # 増強されたデータを保存するためのリスト
    augmented_rows = []

    # 各行ごとにデータ拡張を実施
    for idx, row in train_df.iterrows():
        img = load_img(row['image_path'], target_size=image_size)
        mask = load_img(row['mask_path'], target_size=image_size, color_mode="grayscale")
        
        img_array = img_to_array(img)
        mask_array = img_to_array(mask)
        
        # バッチサイズを1にして、データ拡張
        img_array = np.expand_dims(img_array, axis=0)
        mask_array = np.expand_dims(mask_array, axis=0)
        
        # 同じシードを使用して画像とマスクに同じ変換を適用
        seed = np.random.randint(10000)
        img_gen = image_datagen.flow(img_array, batch_size=1, seed=seed)
        mask_gen = mask_datagen.flow(mask_array, batch_size=1, seed=seed)

        for i in range(num_aug):
            aug_img = next(img_gen)[0]
            aug_mask = next(mask_gen)[0]

            # 保存ファイル名を定義
            new_image_name = f"{row['id']}_{i+1:02d}.jpg"
            new_mask_name = f"{row['id']}_{i+1:02d}.jpg"
            new_image_path = os.path.join(save_to_dir_images, new_image_name)
            new_mask_path = os.path.join(save_to_dir_masks, new_mask_name)
            
            # 拡張画像を保存
            save_img(new_image_path, aug_img)
            save_img(new_mask_path, aug_mask)

            # 新しい行を追加
            augmented_rows.append({
                'id': f"{row['id']}_{i+1:02d}",
                'image_path': new_image_path,
                'mask_path': new_mask_path,
                'original_image_path': row['image_path'],
                'original_mask_path': row['mask_path'],
            })

    # 新しいデータフレームを作成
    aug_train_df = pd.DataFrame(augmented_rows)
    
    return aug_train_df
