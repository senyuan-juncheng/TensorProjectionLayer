import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session
from image_mask_generator import image_mask_generator
from models1 import unet_model, tpl_unet_model
import tensorflow as tf

# Dice LossとIoUメトリクスの定義
import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred, 1)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # 確率を0または1に変換
    intersection = tf.reduce_sum(y_true * y_pred)  # 交差部分
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection  # 結合部分
    return intersection / union

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# トレーニングと評価の関数を定義
def train_and_evaluate_model(df, aug_df, num_epoch, batch_size, num_trials, image_height, image_width, num_channels, leaky_relu, output_dir):
    # トライアルを1からnum_trialsまで繰り返す
    for trial in range(1, num_trials + 1):
        trial_str = f'{trial:02d}'  # 01, 02, 03, ... 30の形式にする

        # テスト結果を保存するフォルダを引数で指定されたoutput_dirに設定
        baseline_history_dir = os.path.join(output_dir, f'baseline/history_{trial_str}.pkl')
        tpl_history_dir = os.path.join(output_dir, f'tpl/history_{trial_str}.pkl')

        os.makedirs(os.path.dirname(baseline_history_dir), exist_ok=True)
        os.makedirs(os.path.dirname(tpl_history_dir), exist_ok=True)

        # ===== データの分割をトライアルごとに変更 ===== #
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=trial, shuffle=True)

        # すでに拡張されたデータから対応する行を抽出
        aug_train_df = aug_df[aug_df['original_image_path'].isin(train_df['image_path']) & aug_df['original_mask_path'].isin(train_df['mask_path'])]

        # 作成済みのdata frameからgeneratorを生成（高さと幅を使用）
        train_gen = image_mask_generator(aug_train_df, batch_size, (image_height, image_width), seed=trial, debug=False)
        test_gen = image_mask_generator(test_df, batch_size, (image_height, image_width), seed=trial, debug=False)

        # ========== ベースラインモデル ========== #
        clear_session()
        baseline_model = unet_model(input_size=(image_height, image_width, 3), c=num_channels)
        baseline_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_coefficient, iou_metric])

        # モデルのトレーニング
        history_baseline = baseline_model.fit(
            train_gen,
            validation_data=test_gen,
            steps_per_epoch=len(aug_train_df) // batch_size,
            validation_steps=len(test_df) // batch_size,
            epochs=num_epoch,  # ここを外部で定義したnum_epochを使用
        )

        # historyをpickle形式で保存
        with open(baseline_history_dir, 'wb') as f:
            pickle.dump(history_baseline.history, f)

        # ========== TPLモデル ========== #
        clear_session()
        tpl_model = tpl_unet_model(input_size=(image_height, image_width, 3), c=num_channels, leaky_relu=leaky_relu)
        tpl_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_coefficient, iou_metric])

        # モデルのトレーニング
        history_tpl = tpl_model.fit(
            train_gen,
            validation_data=test_gen,
            steps_per_epoch=len(aug_train_df) // batch_size,
            validation_steps=len(test_df) // batch_size,
            epochs=num_epoch,  # ここを外部で定義したnum_epochを使用
        )

        # historyをpickle形式で保存
        with open(tpl_history_dir, 'wb') as f:
            pickle.dump(history_tpl.history, f)

        print(f'Trial {trial_str} completed and results saved in {output_dir}.')

