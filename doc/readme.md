# OSS Composs预测成败流程

## 数据预处理data_preporcess

- 将数据处理为本项目便于使用的统一格式。封装，便于后期扩充数据时改用新的读取方式
  
  - {}\_{}.csv，其中包含某个项目的所有metric数据

- 目前的数据来源：
  
  - 每个仓库爬取完成的链接，例：[nodejs仓库的指数分析结果]("https://oss-compass.org/analyze?label=https%3A%2F%2Fgithub.com%2Fnodejs%2Fnode&level=repo")
  
  - 打包来的数据（**是否存入数据库**）

- 输入：仓库名称

- 输出：时间序列、时间戳

- **<u>! Alert !：由于保密需要，原始数据和分析结果不上传GitHub，需要注意配置.gitignore文件和config.ini文</u>件**

  - 新建config.ini，并配置data_path、result_path、 file_list
  - 新建数据目录和结果目录，结果目录的层级为：
    - /raw
    - /segment_data
    - /segment2
- 目前有两个问题项目，最后拼接完成剩余600个项目

![img.png](img.png)![img_1.png](img_1.png)


## 数据处理工具类utils

- 输入和输出格式保持一致，仅进行中间的数据处理

- 切数据段的函数，时间格式<timeline,指数1,指数2,...,指数n>，也就是每个指数的timeline是共通的

- 中间数据要按照上述形式存一下

- 如截取中间有效数据长度、进行数据平滑、切分训练集和测试集等
- 因为这部分可能会产生变动，因此切片后的数据暂时存放在result/segment2中

## 机器学习方法集

- 一系列不同的机器学习方法，包括特征提取等步骤

- 输入：训练集

- 输出：模型

## 评估

- 输入：模型、测试集

- 输出：格式化的评估结果

