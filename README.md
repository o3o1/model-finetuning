电力电子蒸馏数据集管道（本地小模型学习云端大模型知识）。

快速开始

- 1 初始化目录与数据卡片：
  - `python main.py init`
- 2 生成中文种子提示（用于向教师模型提问）：
  - `python main.py make-seed --num 200`
- 3 让云端大模型回答（离线进行），将结果保存为 JSONL，字段包含：`id, response`。
  - 把文件放到 `data/teacher_outputs/teacher.jsonl`
- 4 合并种子与教师输出：
  - `python main.py join --teacher data/teacher_outputs/teacher.jsonl`
- 5 质量过滤：
  - `python main.py quality`
- 6 划分训练/验证/测试集：
  - `python main.py split`

文件结构

- `data/seed/seed_prompts.jsonl`：给教师模型的提问
- `data/teacher_outputs/joined.jsonl`：问题与回答合并
- `data/sft/cleaned.jsonl`：清洗后的配对数据
- `data/sft/train.jsonl|val.jsonl|test.jsonl`：切分后的文件

说明

- 当前代码仅使用标准库，可在无网络环境下准备与清洗数据。
- 教师调用可使用任意云端大模型（需自行离线获取回答），注意合规与授权。
