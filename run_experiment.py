import subprocess
import os
import datetime
import sys
from pathlib import Path
import argparse

def run_command(command, log_file=None):
    """运行命令并记录输出"""
    # 设置环境变量以支持 UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        env=env,
        encoding='utf-8'
    )
    
    # 实时打印并记录输出
    while True:
        try:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # 使用 ascii 编码替换无法显示的字符
                print(output.strip().encode('ascii', 'replace').decode())
                if log_file:
                    log_file.write(output)
                    log_file.flush()
        except UnicodeEncodeError:
            # 如果遇到编码错误，使用 ASCII 替换
            if log_file:
                log_file.write("[Encoding Error - Character skipped]\n")
                log_file.flush()
    
    return process.poll()

def main():
    # 设置控制台输出编码
    if sys.platform == 'win32':
        try:
            subprocess.run(['chcp', '65001'], shell=True)
        except:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.002, help='学习率')
    parser.add_argument('--exp_name', type=str, default='', help='实验名称')
    args = parser.parse_args()
    
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{args.exp_name}_{timestamp}" if args.exp_name else f"exp_{timestamp}"
    exp_dir = f"experiments/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建日志文件
    log_path = os.path.join(exp_dir, "experiment_log.txt")
    metrics_path = os.path.join(exp_dir, "metrics.txt")
    
    with open(log_path, "w", encoding='utf-8') as log_file:
        # 记录实验开始时间和配置
        log_file.write(f"=== Experiment Start Time: {timestamp} ===\n\n")
        
        # 运行训练
        log_file.write("=== Starting Training ===\n")
        train_command = (
            f"python train_jepa.py "
            f"--data_dir "/scratch/DL25SP/train" "
            f"--exp_dir {exp_dir} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--lr {args.lr} "
            f"--min_lr 1e-4"
        )
        log_file.write(f"Training command: {train_command}\n\n")
        
        train_result = run_command(train_command, log_file)
        
        if train_result != 0:
            log_file.write("\nTraining failed!\n")
            return
        
        log_file.write("\n=== Training Complete ===\n\n")
        
        # 运行验证
        log_file.write("=== Starting Evaluation ===\n")
        eval_command = f"python main.py > {metrics_path} 2>&1"
        log_file.write(f"Evaluation command: {eval_command}\n\n")
        
        eval_result = run_command(eval_command, log_file)
        
        if eval_result != 0:
            log_file.write("\nEvaluation failed!\n")
            return
        
        log_file.write("\n=== Evaluation Complete ===\n")
        
        # 读取并记录验证结果
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = f.read()
                log_file.write("\n=== Evaluation Results ===\n")
                log_file.write(metrics)
        
        # 记录实验结束时间
        end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file.write(f"\n=== Experiment End Time: {end_time} ===\n")
        
        # 创建实验总结文件
        summary_path = os.path.join(exp_dir, "experiment_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write("=== Experiment Summary ===\n")
            summary_file.write(f"Start Time: {timestamp}\n")
            summary_file.write(f"End Time: {end_time}\n")
            summary_file.write(f"Training Command: {train_command}\n")
            summary_file.write(f"Evaluation Command: {eval_command}\n\n")
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = f.read()
                    summary_file.write("Evaluation Results:\n")
                    summary_file.write(metrics)

if __name__ == "__main__":
    main() 
