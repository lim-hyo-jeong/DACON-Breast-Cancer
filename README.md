## [Dacon] 유방암의 임파선 전이 예측 AI 경진대회

<b>임효정 (팀명 : lhj) lim.gadi@gmail.com</b>

---

### 📌 대회 설명

1. 주제 : 유방암 병리 슬라이드 영상과 임상 항목을 조합하여 유방암의 임파선 전이 여부 이진 분류
2. 평가 산식 : Macro F1 Score

```python
from sklearn import metrics

score = metrics.f1_score(y_true=true, y_pred=pred, average='macro', sample_weight=sample_weight.values)
```

3. 링크 : https://dacon.io/competitions/official/236011/overview/description

### 📌 개요

1. <b>EDA & Preprocessing</b>
   1. Image Data Preprocessing : Tissue Segmentation, Tissue 위주로 이미지 crop, Tile Creation
   2. Tabular Data Preprocessing : 결측치 처리, Feature Generation
2. <b>Modeling</b>
   1. Multiple Instance Learning : swin_tiny_patch4_window7_224
   2. CNN-Tabular Multi-modal Learning : densenetblur121d + DNN
   3. Tree Based Boosting Algorithm : xgboost
3. <b>Cross Validation</b> : Stratified 5-fold Validation
4. <b>Ensemble</b> : Stacking Ensemble (meta classifier : linear_regression)

### 📌 개발 환경

1. OS : Ubuntu 18.04.6 LTS
2. Python : 3.8.16
3. GPU : A100-SXM4-40GB

### 📌 실행 방법

1. 라이브러리 설치
   <code>pip install -r requirements.txt</code>
2. Preprocessing
   <code>python main.py config/preprocessing_config.yaml preprocessing</code>
3. MIL 모델 training
   <code>python main.py config/mil_config.yaml train</code>
4. MIL 모델 inference
   <code>python main.py config/mil_config.yaml inference</code>
5. CNN-Tabular Multi-modal 모델 training
   <code>python main.py config/convnet_tabular_config.yaml train</code>
6. CNN-Tabular Multi-modal 모델 inference
   <code>python main.py config/convnet_tabular_config.yaml inference</code>
7. XGB 모델 training
   <code>python main.py config/xgb_config.yaml train</code>
8. XGB 모델 inference
   <code>python main.py config/xgb_config.yaml inference</code>
9. Stacking Ensemble (최종 결과물)
   <code>python main.py config/ensemble_config.yaml ensemble</code>

### 📌 파일 구조

```
.
├── ./config
│   ├── ./config/convnet_tabular_config.yaml
│   ├── ./config/ensemble_config.yaml
│   ├── ./config/mil_config.yaml
│   ├── ./config/preprocessing_config.yaml
│   └── ./config/xgb_config.yaml
├── ./data
│   ├── ./data/clinical_info.xlsx
│   ├── ./data/sample_submission.csv
│   ├── ./data/test.csv
│   ├── ./data/test_imgs
│   ├── ./data/test_imgs_crop
│   ├── ./data/test_preprocessed.csv
│   ├── ./data/test_tiles
│   ├── ./data/train.csv
│   ├── ./data/train_imgs
│   ├── ./data/train_imgs_crop
│   ├── ./data/train_preprocessed.csv
│   └── ./data/train_tiles
├── ./ensemble.py
├── ./image_preprocessing.py
├── ./log
├── ./main.py
├── ./metrics.py
├── ./model
│   ├── ./model/densenetblur121d_image1_1024_tabular
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/model_f0_best.pth
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/model_f1_best.pth
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/model_f2_best.pth
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/model_f3_best.pth
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/model_f4_best.pth
│   │   ├── ./model/densenetblur121d_image1_1024_tabular/oof.npy
│   │   └── ./model/densenetblur121d_image1_1024_tabular/preds.npy
│   ├── ./model/swintinypatch4window7_mil16_224
│   │   ├── ./model/swintinypatch4window7_mil16_224/model_f0_best.pth
│   │   ├── ./model/swintinypatch4window7_mil16_224/model_f1_best.pth
│   │   ├── ./model/swintinypatch4window7_mil16_224/model_f2_best.pth
│   │   ├── ./model/swintinypatch4window7_mil16_224/model_f3_best.pth
│   │   ├── ./model/swintinypatch4window7_mil16_224/model_f4_best.pth
│   │   ├── ./model/swintinypatch4window7_mil16_224/oof.npy
│   │   └── ./model/swintinypatch4window7_mil16_224/preds.npy
│   └── ./model/xgboost_tabular
│       ├── ./model/xgboost_tabular/model_f0_best.pkl
│       ├── ./model/xgboost_tabular/model_f1_best.pkl
│       ├── ./model/xgboost_tabular/model_f2_best.pkl
│       ├── ./model/xgboost_tabular/model_f3_best.pkl
│       ├── ./model/xgboost_tabular/model_f4_best.pkl
│       ├── ./model/xgboost_tabular/oof.npy
│       └── ./model/xgboost_tabular/preds.npy
├── ./README.md
├── ./requirements.txt
├── ./settings.py
├── ./submission
├── ./tabular_preprocessing.py
├── ./torch_dataset.py
├── ./torch_model.py
├── ./torch_trainer.py
├── ./transforms.py
├── ./utils.py
└── ./xgb_trainer.py
```
