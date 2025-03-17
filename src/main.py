import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from db_connector import DatabaseConnector
from data_diff import DataDiffer
from llm_analyzer import LLMAnalyzer

def main():
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据库数据自动化对比工具')
    parser.add_argument('--db1', required=True, help='第一个数据库的连接字符串')
    parser.add_argument('--db2', required=True, help='第二个数据库的连接字符串')
    parser.add_argument('--table', required=True, help='要比较的表名')
    parser.add_argument('--keys', required=True, help='用于匹配行的键列，逗号分隔')
    parser.add_argument('--columns', help='要比较的列，逗号分隔，默认为所有列')
    parser.add_argument('--where', help='WHERE子句，应用于两个数据库')
    parser.add_argument('--limit', type=int, help='限制每个数据库返回的行数')
    parser.add_argument('--output', help='输出结果的文件路径')
    parser.add_argument('--no-llm', action='store_true', help='禁用LLM分析')
    
    args = parser.parse_args()
    
    # 连接数据库并获取数据
    db1 = DatabaseConnector(args.db1)
    db2 = DatabaseConnector(args.db2)
    
    columns = args.columns.split(',') if args.columns else None
    key_columns = args.keys.split(',')
    
    print(f"从 {args.db1} 获取数据...")
    df1 = db1.get_table_data(args.table, columns, args.where, args.limit)
    
    print(f"从 {args.db2} 获取数据...")
    df2 = db2.get_table_data(args.table, columns, args.where, args.limit)
    
    # 比较数据
    print("比较数据差异...")
    differ = DataDiffer()
    diff_results = differ.compare_dataframes(df1, df2, key_columns)
    
    # 格式化差异结果
    formatted_diff = differ.format_diff_results(
        diff_results, 
        df1_name=f"数据库1 ({args.db1})", 
        df2_name=f"数据库2 ({args.db2})"
    )
    
    print("\n数据差异摘要:")
    print(formatted_diff)
    
    # 使用LLM分析差异（如果未禁用）
    if not args.no_llm:
        try:
            print("\n使用LLM分析差异...")
            # 检查API密钥是否已设置
            if not LLMAnalyzer.check_api_key():
                print("警告: 无法使用LLM分析，因为未设置DEEPSEEK_API_KEY")
                print("请设置环境变量或在.env文件中配置后重试")
            else:
                analyzer = LLMAnalyzer()
                context = {
                    "数据库1": args.db1,
                    "数据库2": args.db2,
                    "表名": args.table,
                    "比较列": args.columns if args.columns else "所有列",
                    "WHERE条件": args.where if args.where else "无"
                }
                analysis = analyzer.analyze_differences(formatted_diff, context)
                print("\nLLM分析结果:")
                print(analysis)
        except Exception as e:
            print(f"LLM分析失败: {str(e)}")
    
    # 保存结果到文件（如果指定）
    if args.output:
        print(f"\n保存结果到 {args.output}...")
        output_path = args.output
        # 确保输出文件以.html结尾
        if not output_path.lower().endswith('.html'):
            output_path += '.html'
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n<head>\n")
            f.write("<meta charset='utf-8'>\n")
            f.write("<title>数据差异报告</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1 { color: #2c3e50; }\n")
            f.write("h2 { color: #3498db; margin-top: 30px; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin-top: 20px; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write(".diff-added { background-color: #d4edda; }\n")
            f.write(".diff-removed { background-color: #f8d7da; }\n")
            f.write(".diff-changed { background-color: #fff3cd; }\n")
            f.write("</style>\n</head>\n<body>\n")
            
            f.write("<h1>数据差异报告</h1>\n\n")
            f.write("<h2>比较信息</h2>\n<ul>\n")
            f.write(f"<li><strong>数据库1:</strong> {args.db1}</li>\n")
            f.write(f"<li><strong>数据库2:</strong> {args.db2}</li>\n")
            f.write(f"<li><strong>表名:</strong> {args.table}</li>\n")
            f.write(f"<li><strong>比较列:</strong> {args.columns if args.columns else '所有列'}</li>\n")
            f.write(f"<li><strong>WHERE条件:</strong> {args.where if args.where else '无'}</li>\n</ul>\n")
            
            f.write("<h2>数据差异摘要</h2>\n")
            # 将纯文本差异转换为HTML格式
            html_diff = formatted_diff.replace('\n', '<br>').replace(' ', '&nbsp;')
            f.write(f"<pre>{html_diff}</pre>\n")
            
            if not args.no_llm and 'analysis' in locals():
                f.write("<h2>LLM分析结果</h2>\n")
                html_analysis = analysis.replace('\n', '<br>')
                f.write(f"<div>{html_analysis}</div>\n")
            elif not args.no_llm:
                f.write("<h2>LLM分析结果</h2>\n")
                f.write("<div>LLM分析失败</div>\n")
                
            f.write("</body>\n</html>")
        
        print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    main()