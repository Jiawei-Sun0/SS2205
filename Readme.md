# スキルゼミ2022 Python深層学習 喉頭内視鏡画像の声帯と声門抽出

U-Netによる喉頭内視鏡画像の声帯(声門)抽出。セグメンテーションタスク用のカスタムDatasetクラスを使用し、one-channelバージョンの損失関数はDice loss、two-channelバージョンはDice loss+Cross entropy lossを使用しています。

## データ

- 元画像：png画像(RGBカラー、8 bitグレースケールどちらも対応)
    - 学習、テストデータとも画像ファイルのチャンネル数より `Unet` モデルの入力チャンネル数を自動設定するようにしています。

- マスク画像：png画像(RGBカラー、8 bitグレースケールどちらも対応)
    - グレースケール画像で画素値が1以上の画素をラベルとする（RGBカラーの場合はグレースケール画像に自動変換する）。

### 画像の配置方法
- `image` (元画像用)、`mask` (マスク画像用) サブフォルダを作成し、それぞれの画像を配置する。元画像とマスク画像のファイル名は同じです。

       (data path)
         |- image
         |    |- image000.png
         |    |- image001.png
         |    |- image002.png
         |
         |- mask
              |- image000.png
              |- image001.png
              |- image002.png

## ネットワーク

- 本課題に使用したモデル：
  - U-Net　[Ronneberger O , MICCAI 2015 : 234-241]
  - Attention U-Net　[Ozan O, arXiv 2018 :1804.03999]
  - SEブロックを加えたAttention U-Net　[Hu J, CVPR 2018 : 7132-7141]
  - U-Net++　[Zhou Z, DLMIP 2018 : 3-11]
- 基本的な使い方は以下の通りです。実行するとき渡された引数よりモデルを選択する。

      if model == 0:
        model = Unet(in_channels, out_channels, first_filter_num)
      elif model == 1:
          model = attUnet(in_channels, out_channels, first_filter_num)
      elif model == 2:
          model = attSEunet(in_channels, out_channels, first_filter_num)
      elif model == 3:
          model = NestedUNet(in_channels, out_channels, first_filter_num)

     - in_channels: 入力チャンネル数(グレースケール画像の場合は1, RGBカラー画像の場合は3)
     - out_channels: 出力チャンネル(基本的には1、複数種類の領域を抽出したい場合はカスタムデータセット、損失関数などの修正が必要です)。
     - first_filter_num: 最初のconvolutionのフィルタ数(論文値は64)。この数を変えるとモデルの規模(=消費GPUメモリ量)が変化します。


## 学習方法

学習は `training_segmentation.py`もしくは`twoCH_training_segmentation.py` を使用します。コマンドの使い方は以下の通りです。

     $ python training_segmentation.py [学習用データのパス名] [検証用データのパス名] [出力のパス名] (オプション) 
     $ python twoCH_training_segmentation.py [学習用データのパス名] [検証用データのパス名] [出力のパス名] (オプション) 

- オプションについて
  - `--gpu_id (-g)`: GPU ID。複数GPU搭載のマシンを使用しない場合は設定不要(デフォルトの0が指定)。詳細は各研究室の先輩、教員に聞いてください。
  - `--first_filter_num (-f)`: 最初のconvolutionのフィルタ数(デフォルト:16)。詳細は上記参照
    - この値を変更した場合、テスト(推論)用のコードでも指定する必要があるので、記録をしてください。
  - `--learning_rate (-l)`: 学習率(デフォルト:0.001)    
  - `--beta_1 (-l)`: Adam(Optimizer)のパラメータbeta_1(デフォルト:0.99、基本的には 0.9 - 0.99 の値を使用します)。      
  - `--batch_size (-b)`: バッチサイズ(デフォルト:8)
  - `--max_epoch_num (-m)`: 学習(最大)エポック数(デフォルト:50)
  - `--time_stamp`: 出力されるファイル名に使われるタイムスタンプです。未指定の場合はプログラム開始時の時刻(YYYYMMDDhhmmss形式)が記載されます。
  - `--model (-mo)`: どのモデルを選択する。（デフォルト：0 —> U-Net）
  - `--augmentation (-a)`: データ拡張をするかどうか。（デフォルト：0）
  

- 出力ファイル
  - loss_log_{time_stamp}_model:{model}_aug:{augmentation}(_2ch).csv
  - model_best_{time_stamp}_model:{model}_aug:{augmentation}(_2ch).pth

## テスト方法

テストは `test_segmentation.py`もしくは`twoCH_test_segmentation.py` を使用します。コマンドの使い方は以下の通りです。

     $ python test_segmentation.py [テスト用データのパス名] [モデルのファイル名] [出力のパス名] (オプション) 
     $ python twoCH_test_segmentation.py [テスト用データのパス名] [モデルのファイル名] [出力のパス名] (オプション)

- オプションについて
  - `--gpu_id (-g)`: GPU ID。複数GPU搭載のマシンを使用しない場合は設定不要(デフォルトの0を指定)。詳細は各研究室の先輩、教員に聞いてください。
  - `--first_filter_num (-f)`: 最初のconvolutionのフィルタ数(デフォルト:16)。学習時にこの値を変更した場合は指定してください。
  - `--binarize_threshold (-t)`: U-Net出力からマスク画像を取得する際の閾値処理の閾値(デフォルト:0.5) 。
  - `--time_stamp`: 出力されるファイル名に使われるタイムスタンプです。未指定の場合はプログラム開始時の時刻(YYYYMMDDhhmmss形式)が記載されます。
  - `--export_mask`: 出力マスク画像(閾値処理後)を出力する場合はこのオプションを指定してください。
  - `--model (-mo)`: モデルファイルのネットワーク。（デフォルト：0 —> U-Net）

- 出力ファイル
  - `test_result_th{閾値}_{time stamp}.csv`
    - テストに使用した各データの出力マスクと正解マスクとのDice係数が記載されています。最初の列のインデックスは出力マスク画像のインデックスと対応しています。
  - `mask_th{閾値}_{time stamp}_{index}.png` (export_maskオプションを指定した場合)
    - 出力マスク画像(閾値処理後)で、マスク部分の画素値は255です。

## フォルダ内のファイルの説明

- `att_SE_Unet.py`: SEブロックを加えたAttention U-Netモデルのクラスファイルです。
- `attUnet.py`: Attention U-Netモデルのクラスファイルです。
- `dataset_seg_aug.py`: data augmentationを実装したカスタムデータセットのファイルです。
- `dataset_segmentation.py`: セグメンテーション用カスタムデータセットクラスのファイルです。
- `dice.py`: Dice係数計算用のクラス(PyTorch)、関数(numpy)が書かれています。
- `diceCE.py`: DiceとCrossEntropyLossを組み合わせたloss関数です。
- `overlay.py`: alpha blending用関数です。
- `test_segmentation.py`: テスト(推論)用コードです。
- `training_segmentation.py`: 学習用コードです。
- `twoCH_dataset_aug.py`: two-channelsとdata augmentation対応したカスタムデータセットクラスのファイルです。
- `twoCH_test_segmentation.py`: two-channelsのテスト(推論)用コードです。
- `twoCH_training_segmentation.py`: two-channelsの学習用コードです。
- `Unet_plus.py`: U-Net++モデルのクラスファイルです。
- `unet.py`: U-Netモデルのクラスファイルです。

## 主に実装した事項

1. マスク画像への後処理
  - ラベリングを用いた最大領域取得を加えた。

2. ハイパーパラメータ探索
  - ランダムサーチを使って各モデルを20日試行した。

3. Data augmentation
  - シフトや回転、スケールを基づいたデータ拡張を実装した。

4. Softmax
　- Softmaxレイヤーを使って２チャンネルにおいて同じ領域を同時抽出を防いだ。


