import os
import shutil

def remove_pycache_folders(root_folder):
    """
    递归删除项目文件夹中的所有__pycache__文件夹
    """
    for foldername in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, foldername)
        
        if foldername == '__pycache__' and os.path.isdir(folder_path):
            # 如果是__pycache__文件夹，则删除它
            print(f"删除文件夹: {folder_path}")
            shutil.rmtree(folder_path)
        elif os.path.isdir(folder_path):
            # 如果是文件夹，则递归处理
            remove_pycache_folders(folder_path)

if __name__ == "__main__":
    project_folder = os.getcwd()  # 获取当前工作目录
    remove_pycache_folders(project_folder)
    print("已删除所有__pycache__文件夹")
