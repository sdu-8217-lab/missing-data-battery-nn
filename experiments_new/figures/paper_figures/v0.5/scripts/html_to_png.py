"""
将 HTML 文件转换为 PNG 图片
需要安装 selenium 和 Chrome WebDriver
"""
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time


def html_to_png(html_path: Path, output_path: Path, width: int = 1400, height: int = 900):
    """将 HTML 文件渲染为 PNG 图片"""
    
    # 设置 Chrome 选项
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(f'--window-size={width},{height}')
    
    # 启动浏览器
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # 加载 HTML 文件
        html_url = html_path.resolve().as_uri()
        driver.get(html_url)
        
        # 等待页面渲染
        time.sleep(2)
        
        # 截图
        driver.save_screenshot(str(output_path))
        print(f"✅ 已保存: {output_path}")
        
    finally:
        driver.quit()


def main():
    parser = argparse.ArgumentParser(description='Convert HTML to PNG')
    parser.add_argument('--html', type=Path, default=Path('figure6_experimental_flow.html'),
                        help='Input HTML file path')
    parser.add_argument('--output', type=Path, default=Path('../experimental_flow.png'),
                        help='Output PNG file path')
    parser.add_argument('--width', type=int, default=1400, help='Viewport width')
    parser.add_argument('--height', type=int, default=900, help='Viewport height')
    
    args = parser.parse_args()
    
    html_to_png(args.html, args.output, args.width, args.height)


if __name__ == '__main__':
    main()
