import os
import yaml

def create_directory_structure_dict(folder_path):
    """
    创建文件夹的目录结构字典
    """
    directory_structure = {}
    for root, dirs, files in os.walk(folder_path):
        current_dir = directory_structure
        # 将目录路径拆分为列表
        dir_parts = os.path.relpath(root, folder_path).split(os.path.sep)
        
        for part in dir_parts:
            current_dir = current_dir.setdefault(part, {})
        
        # 将文件添加到当前目录
        for file in files:
            current_dir[file] = {}
    
    return directory_structure

def save_directory_structure_to_yaml(directory_structure, output_file):
    """
    将文件夹的目录结构保存为YAML格式文件
    """
    with open(output_file, 'w') as yaml_file:
        yaml.dump(directory_structure, yaml_file, default_flow_style=False)

if __name__ == "__main__":
    folder_path = os.getcwd()  # 指定您的文件夹路径
    output_file = "directory_structure.yaml"  # 指定输出的YAML文件名

    directory_structure = create_directory_structure_dict(folder_path)
    save_directory_structure_to_yaml(directory_structure, output_file)
    print(f"目录结构已保存到 {output_file}")
