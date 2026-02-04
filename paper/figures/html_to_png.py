# -*- coding: utf-8 -*-
"""
将HTML文件转换为PNG图像
使用html2image库 + Edge浏览器
"""
import os
import sys

from html2image import Html2Image

def convert_html_to_png(html_file, output_file=None, size=(1200, 800)):
    """将HTML文件转换为PNG"""
    if output_file is None:
        output_file = html_file.replace('.html', '.png')
    
    # 获取绝对路径
    html_path = os.path.abspath(html_file)
    output_dir = os.path.dirname(html_path)
    
    print(f"Converting {html_file} -> {output_file}")
    print(f"Size: {size}")
    
    try:
        # 使用Edge浏览器
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        
        hti = Html2Image(
            output_path=output_dir,
            size=size,
            browser_executable=edge_path,
            custom_flags=[
                '--default-background-color=ffffff',
                '--hide-scrollbars',
                '--disable-gpu',
                '--no-sandbox',
            ]
        )
        
        hti.screenshot(
            html_file=html_path,
            save_as=output_file
        )
        
        print(f"[OK] Success: {output_file}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        return False

def main():
    # 定义要转换的文件和对应尺寸
    files = [
        ('fig1_problem_scenario.html', (1000, 500)),
        ('fig2_baseline_vs_mim.html', (1000, 550)),
        ('fig3_model_architectures.html', (1150, 550)),
        ('fig4_joint_training.html', (1050, 500)),
        ('fig6_model_selection.html', (1050, 450)),
        ('fig7_performance_params.html', (850, 550)),
    ]
    
    success_count = 0
    
    for html_file, size in files:
        if os.path.exists(html_file):
            if convert_html_to_png(html_file, size=size):
                success_count += 1
        else:
            print(f"[NOT FOUND] File not found: {html_file}")
    
    print(f"\n{'='*50}")
    print(f"Converted: {success_count}/{len(files)} files")
    print(f"{'='*50}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
