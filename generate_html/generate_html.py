from jinja2 import Environment, FileSystemLoader
import os


def generate_html():
    env = Environment(loader=FileSystemLoader('generate_html/template'))
    template = env.get_template('simple.html')
    data = {
        'page_title': '测试结果',#这是页面的标题
        'main_title': '测试结果',#这是页内的标题
        'description': '以下为本次测试的结果汇总',#（可选）添加总体文字描述
        'folders_data': []
    }

    root_dir = 'data/html/'
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            images = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(folder_path, filename)
                    images.append({'path': '../../' + image_path, 'title': filename})
            data['folders_data'].append({'folder_name': folder_name, 'images': images})

    html_output = template.render(data)
    if not os.path.exists('generate_html/outputs'):
        os.makedirs('generate_html/outputs')
    with open('generate_html/outputs/output.html', 'w', encoding='utf-8') as output_file:
        output_file.write(html_output)
