import pandas as pd
from dataset import calculate_similarity

# 读取模板 CSV 文件和目标 CSV 文件
template_file = './metadata/data_test.csv' # 模板CSV文件路径
target_file = './metadata/data_0920_i.csv' # 目标CSV文件路径

template_df = pd.read_csv(template_file)
target_df = pd.read_csv(target_file)

# 假设我们要使用 'template_column' 列作为模板中的列，'target_column' 列作为目标表格中的列
template_column = 'Seq'  # 替换为你的模板表格列名
target_column = 'Seq'      # 替换为你的目标表格列名

# 创建一个空的 DataFrame，用于存储符合条件的行，并复制目标表格的表头
extracted_rows = pd.DataFrame(columns=target_df.columns)

# 遍历目标表格的每一行
for index, target_row in target_df.iterrows():
    target_value = target_row[target_column]
    
    # 遍历模板表格的每个值
    for template_value in template_df[template_column]:
        
        # 计算指标
        metric = calculate_similarity(target_value.strip().upper(), template_value.strip().upper())
        
        # 如果指标大于0.3，则提取目标表格的该行
        if metric > 0.3:
            extracted_rows = pd.concat([extracted_rows, pd.DataFrame([target_row])], ignore_index=True)
            # 只要找到一个匹配值，就可以继续检查下一行
            break

# 将提取的行保存到一个新的CSV文件中，确保表头与目标表格一致
output_file = './metadata/data_simi.csv'
extracted_rows.to_csv(output_file, index=False)

print(f"符合条件的行已保存到 {output_file}")