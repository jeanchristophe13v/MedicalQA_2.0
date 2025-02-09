from chat_agent import ChatAgent
import sys
import time
import threading
import itertools
from colorama import init, Fore, Style  # 添加颜色支持
import os
import random

# 初始化颜色支持匹配
init()

def loading_animation():
    """加载动画"""
    # 更丰富的加载动画帧
    frames = [
        "🤔 Loading... ",
        "💭 Loading... ",
        "💡 Loading... ",
        "✨ Loading... "
    ]
    dots = ["", ".", "..", "..."]
    i = 0
    while loading_animation.running:
        frame = frames[i % len(frames)]
        dot = dots[i % len(dots)]
        sys.stdout.write('\r' + Fore.BLUE + frame + dot + Style.RESET_ALL)
        sys.stdout.flush()
        time.sleep(0.3)
        i += 1

def start_loading():
    loading_animation.running = True
    threading.Thread(target=loading_animation).start()

def stop_loading():
    loading_animation.running = False
    sys.stdout.write('\r' + ' ' * 20 + '\r')  # 清除加载动画

def stream_output(text):
    """直接输出完整文本"""
    print(text)

def print_welcome():
    """打印欢迎信息"""
    welcome_text = """
╭────────────────────────────────────────╮
│  - 输入问题即可开始对话                │
│  - 输入 'q' 退出程序                   │
│  - 输入 'clear' 清屏                   │
╰────────────────────────────────────────╯
"""
    print(Fore.GREEN + welcome_text + Style.RESET_ALL)

def animate_generating():
    """生成动画"""
    frames = ["⋮", "⋯", "⋰", "⋱"]
    i = 0
    message = "思考中"
    
    while animate_generating.running:
        # 完整清理前一帧
        sys.stdout.write('\r' + ' ' * 50)
        sys.stdout.flush()
        
        
        # 显示新帧
        current_frame = f"\r{frames[i % len(frames)]} {message}"
        if i % 4 == 0:
            current_frame += "."
        elif i % 4 == 1:
            current_frame += ".."
        elif i % 4 == 2:
            current_frame += "..."
            
        sys.stdout.write(current_frame)
        sys.stdout.flush()
        time.sleep(0.3)
        i += 1

def start_generating_animation():
    """开始生成回答动画"""
    animate_generating.running = True
    threading.Thread(target=animate_generating).start()

def stop_generating_animation():
    """停止并清理动画"""
    animate_generating.running = False
    time.sleep(0.1)  # 等待线程完成
    # 彻底清理动画
    sys.stdout.write('\r' + ' ' * 50 + '\r')
    sys.stdout.flush()

def main():
    try:
        print_welcome()
        agent = None
        
        start_loading()
        try:
            agent = ChatAgent("data")
        finally:
            stop_loading()
            
        while True:
            query = input("\n请输入问题: ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
                
            if not query:
                continue
                
            animate_generating.running = True
            thread = threading.Thread(target=animate_generating)
            thread.daemon = True
            thread.start()
            
            try:
                for response in agent.chat(query):
                    stop_generating_animation()
                    if response:
                        # 使用打字机效果输出
                        stream_output(f"\n{response}")
            except Exception as e:
                stop_generating_animation()
                print(f"\n错误: {str(e)}")
                
    except Exception as e:
        print(f"\n程序初始化失败: {str(e)}")
    finally:
        print("\n感谢使用！再见！")

if __name__ == "__main__":
    # 检查是否在 Windows 终端中运行
    if os.name == 'nt':
        os.system('color')
    main()