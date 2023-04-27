# Adenocyte-classification-detection
## 数据情况
### 原始数据
医生提供了三批数据

1. 只标了异常腺细胞，一共3048个标注，保存在/remote-home/share/DATA/RedHouse/Adenocyte/xmlfile_0118
2. 标了五类细胞，包括正常细胞一共4374个标注，保存在~/Adenocyte/xmlfile_0327
<img width="351" alt="image" src="https://user-images.githubusercontent.com/83853473/233537843-9737c29b-5bb7-49ab-9d9f-4cda43b922a4.png">

3. 正常细胞的WSI，无标注，保存在/remote-home/share/DATA/RedHouse/AdenocyteBag/SVSfile0413

### 数据集
0. 二分类数据集，基于第1批原始数据构建，保存在~/AdenocyteBag/abmil_0118data_512_128

   包含1912个正包和12984个负包
1. 二分类数据集，基于第1批原始数据构建，保存在~/AdenocyteBag/abmil_0118data_768_448
   
   包含5798个正包和10783个负包

2. 二分类数据集，基于第1批原始数据得到的正包+从第3批原始数据中采样得到的负包，正包和负包数量都为5798个，保存在~/AdenocyteBag/abmil_0413data_768
3. 五分类数据集，基于第2批原始数据构建，保存在~/AdenocyteBag/abmil_0326data_512_180
4. Faster R-CNN数据集，基于第2批和第3批原始数据构建，包含4856个阳性图片和1000个阴性图片，训练集：测试集=7：3，训练集包含4560个阳性bbox和9566个阴性bbox，测试集包含1990个阳性bbox和3856个阴性bbox

5. 二分类数据集，基于第2批和第3批原始数据构建，包含4856个阳性图片和4856个阴性图片，保存在~/AdenocyteBag/resnet_data_768_384
## 代码
- abmil_2type: ABMIL二分类
- abmil_5type: ABMIL五分类
- resnet_2type: ResNet二分类
- resnet_5type: ResNet五分类
- data_process: 数据处理
- dataset_split: 分割数据集

## ABMIL
### 第1个数据集（加正常细胞前）
|lr|epochs|Acc|Precision|Recall|F1-score|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1e-6 | 27 | 0.775 | 0.651 | 0.656 | 0.653 |


### 第2个数据集（加正常细胞后）

|lr|epochs|Acc|Precision|Recall|F1-score|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1e-5 | 10 | 0.972 | 0.977 | 0.966 | 0.971 |

## ResNet
### 第0个数据集（加正常细胞前）
|lr|epochs|Acc|Precision|Recall|F1-score|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 5e-4 | 26 | 0.9049 | 0.64 | 0.58 | 0.61 |

### 第2个数据集

### 第5个数据集（加正常细胞后）
|lr|epochs|Acc|Precision|Recall|F1-score|
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1e-5 | 20 | 1 | 1 | 1 | 1 |
| 1e-5 | 20 | 0.9999 | 0.9990 | 1 | 0.9995 |


混淆矩阵

970 1

0 971
## Faster R-CNN
代码下载在/remote-home/share/mmdetection-main
