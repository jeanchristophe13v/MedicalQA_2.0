import sys
import time
import threading
import os
from colorama import init, Fore, Style  # 仅保留必要的初始导入

# 初始化颜色支持匹配
init()

class LoadingAnimation:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        """开始加载动画"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """停止加载动画"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.2)
            # 清除动画
            sys.stdout.write('\r' + ' ' * 30 + '\r')
            sys.stdout.flush()

    def _animate(self):
        """加载动画实现"""
        dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self.running:
            sys.stdout.write('\r' + dots[i % len(dots)] + ' Loading...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

# 创建全局loading动画实例
loading_animation = LoadingAnimation()

def clear_loading_line():
    """清除loading残留的文本"""
    sys.stdout.write('\r' + ' ' * 30 + '\r')
    sys.stdout.flush()

def print_with_loading_clear(text):
    """打印文本前先清除loading残留，不自动重启动画"""
    clear_loading_line()
    print(text)

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
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
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
    # 立即显示欢迎信息
    print_welcome()
    
    specific_files = sys.argv[1:] if len(sys.argv) > 1 else None
    
    try:
        # 延迟导入ChatAgent，这样不会阻塞欢迎信息的显示
        from chat_agent import ChatAgent
        
        if specific_files:
            print_with_loading_clear(f"准备加载指定的PDF文件: [{', '.join(specific_files)}]")
        else:
            print_with_loading_clear("准备加载data文件夹中的所有PDF文件")
        
        loading_animation.start()
        try:
            agent = ChatAgent("data", specific_files)
        finally:
            loading_animation.stop()

        while True:
            query = input("\n请输入问题: ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            elif query.lower() == 'update':  # 添加 update 命令
                loading_animation.start()
                try:
                    agent.update_knowledge_base("data")
                finally:
                    loading_animation.stop()
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