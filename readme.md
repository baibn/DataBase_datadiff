## 使用示例

```
### 基本比较
比较两个 SQLite 数据库中的 users 表，使用 id 列作为键：

```bash
python /Users/******/baibn/Data_diff/src/main.py --db1 sqlite:///prod.db --db2 sqlite:///test.db --table users --keys id
 ```

```

### 指定列和条件
只比较特定列并添加 WHERE 条件：

```bash
python /Users/******/baibn/Data_diff/src/main.py --db1 sqlite:///prod.db --db2 sqlite:///test.db --table orders --keys order_id --columns customer_id,total,status --where "created_at > '2023-01-01'"
 ```

```

### 生成HTML报告
将比较结果保存为HTML报告：

```bash
python /Users/******/baibn/Data_diff/src/main.py --db1 sqlite:///prod.db --db2 sqlite:///test.db --table products --keys product_id --output /Users/******/baibn/Data_diff/reports/product_diff_report.html
 ```

```

### 禁用LLM分析
如果不需要大模型分析，可以添加 --no-llm 标志：

```bash
python /Users/******/baibn/Data_diff/src/main.py --db1 sqlite:///prod.db --db2 sqlite:///test.db --table users --keys id --no-llm
 ```

```

## 注意事项
1. 在运行程序前，确保已设置 DEEPSEEK_API_KEY 环境变量或在 .env 文件中配置它（如果需要使用LLM分析）
2. 数据库连接字符串格式取决于您使用的数据库类型，请确保格式正确
3. 输出的HTML报告会自动添加 .html 扩展名（如果没有指定）
4. 程序会在控制台输出执行进度和摘要信息
这个工具可以帮助您快速比较两个数据库中表的差异，并通过大模型分析提供洞察，非常适合数据迁移验证、环境对比等场景。