import os
import pandas as pd
from datasets import load_dataset

# 设置镜像（解决网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 方法：直接指定 arrow 文件的 URL（跳过 .py 脚本）
data_files = {
    'train': 'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-train.arrow',
    'test': 'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-test.arrow',
    'validation': 'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-validation.arrow'
}

print("正在从镜像下载 Arrow 文件...")
dataset = load_dataset('arrow', data_files=data_files)

# 合并训练集和测试集
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])
df = pd.concat([df_train, df_test], ignore_index=True)

# 重命名列
df = df.rename(columns={'text': 'review'})

print(f"下载完成！共 {len(df)} 条数据")
print(f"好评: {len(df[df['label']==1])}")
print(f"差评: {len(df[df['label']==0])}")

# 保存为 CSV
df[['review', 'label']].to_csv('large_data.csv', index=False, encoding='utf-8-sig')
print("已保存到 large_data.csv")