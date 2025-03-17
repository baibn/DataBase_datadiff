import os
import sys
import time
from typing import Any, Dict, List, Optional

import openai
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class LLMAnalyzer:
    """使用LLM分析数据差异"""
    
    @staticmethod
    def check_api_key():
        """
        检查是否设置了SILICONFLOW_API_KEY环境变量
        如果未设置，尝试从.env文件加载，并提供友好的错误信息
        """
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            # 尝试从项目根目录的.env文件加载
            env_paths = [
                os.path.join(os.getcwd(), '.env'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    # 尝试再次加载.env文件
                    load_dotenv(env_path)
                    api_key = os.getenv("SILICONFLOW_API_KEY")
                    if api_key:
                        print(f"已从 {env_path} 加载 SILICONFLOW_API_KEY")
                        break
            
            if not api_key:
                print("错误: 未找到 SILICONFLOW_API_KEY 环境变量")
                print("请通过以下方式之一设置 SILICONFLOW_API_KEY:")
                print("1. 在环境变量中设置 SILICONFLOW_API_KEY")
                print("2. 在项目根目录创建 .env 文件并添加: SILICONFLOW_API_KEY=your_api_key_here")
                return False
        
        return True
    
    @staticmethod
    def load_env_config():
        """
        从环境变量加载所有LLM配置参数
        
        Returns:
            包含配置参数的字典
        """
        config = {}
        
        # 只加载硅基流动的API密钥
        config["api_key"] = os.getenv("SILICONFLOW_API_KEY")
        if not config["api_key"]:
            print("警告: 未设置SILICONFLOW_API_KEY环境变量")
        
        # 只使用硅基流动的API基础URL
        api_base = os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
        config["api_base"] = api_base
            
        # 加载模型配置参数
        model_name = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-ai/DeepSeek-V2.5")
        config["model_name"] = model_name
            
        temperature = os.getenv("SILICONFLOW_TEMPERATURE")
        if temperature:
            try:
                config["temperature"] = float(temperature)
            except ValueError:
                print(f"警告: 环境变量SILICONFLOW_TEMPERATURE值'{temperature}'无法转换为浮点数，使用默认值")
                
        max_tokens = os.getenv("SILICONFLOW_MAX_TOKENS")
        if max_tokens:
            try:
                config["max_tokens"] = int(max_tokens)
            except ValueError:
                print(f"警告: 环境变量SILICONFLOW_MAX_TOKENS值'{max_tokens}'无法转换为整数，使用默认值")
        
        return config
    
    def __init__(self, api_key: str = None, api_base: str = None, model_name: str = None, 
                 temperature: float = None, max_tokens: int = None, retry_count: int = 3):
        """
        初始化LLM分析器
        
        Args:
            api_key: 硅基流动API密钥，如果为None，则从环境变量中获取
            api_base: 硅基流动API基础URL，如果为None，则从环境变量中获取
            model_name: 使用的模型名称，如果为None，则从环境变量中获取，默认为"deepseek-ai/DeepSeek-V2.5"
            temperature: 生成文本的随机性，值越高随机性越大，默认为0.7
            max_tokens: 生成文本的最大长度，默认为2000
            retry_count: 连接失败时的重试次数，默认为3
        """
        # 从环境变量加载配置
        env_config = self.load_env_config()
        
        # 检查API密钥
        if api_key is None:
            api_key = env_config.get("api_key")
            if not api_key:
                raise ValueError("必须提供API密钥或设置SILICONFLOW_API_KEY环境变量")
        
        # 使用参数值或环境变量值，如果都没有则使用默认值
        if api_base is None:
            api_base = env_config.get("api_base", "https://api.siliconflow.cn/v1/chat/completions")
            
        if model_name is None:
            model_name = env_config.get("model_name", "deepseek-ai/DeepSeek-V2.5")
            
        if temperature is None:
            temperature = env_config.get("temperature", 0.7)
            
        if max_tokens is None:
            max_tokens = env_config.get("max_tokens", 2000)
        
        # 保存重试次数
        self.retry_count = retry_count
        
        # 保存API参数
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 保存配置以便后续可能的调整
        self.config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_base": api_base
        }
        
        # 初始化OpenAI客户端
        openai.api_key = self.api_key
        # 如果使用的是自定义API基础URL，设置base_url
        if api_base and not api_base.endswith("/chat/completions"):
            openai.base_url = api_base
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试与LLM的连接"""
        try:
            # 简单的测试查询
            self._call_llm_api("测试连接")
            return True
        except Exception as e:
            print(f"连接测试失败: {str(e)}")
            return False
    
    def _call_llm_api(self, prompt, system_message="你是一位数据分析专家"):
        """调用LLM API"""
        try:
            # 使用新版本OpenAI客户端
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if not self.api_base.endswith("/chat/completions") else self.api_base.rsplit("/chat/completions", 1)[0]
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            raise
    
    def _ensure_connection(self):
        """确保LLM连接可用，如果不可用则尝试重新连接"""
        for attempt in range(self.retry_count):
            try:
                if self._test_connection():
                    return True
            except Exception as e:
                print(f"连接失败: {str(e)}")
                if attempt < self.retry_count - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        print(f"在 {self.retry_count} 次尝试后仍无法连接LLM")
        return False
    
    def analyze_differences(self, diff_text: str, context: Dict[str, Any] = None, 
                           custom_prompt: Optional[str] = None) -> str:
        """
        使用LLM分析数据差异
        
        Args:
            diff_text: 格式化的差异文本
            context: 提供给LLM的额外上下文信息
            custom_prompt: 自定义提示模板，如果为None则使用默认模板
            
        Returns:
            LLM生成的分析结果
        """
        # 确保连接可用
        if not self._ensure_connection():
            raise ConnectionError("无法连接到LLM服务。请检查网络连接、API密钥和API基础URL是否正确。")
        
        # 创建提示模板
        if custom_prompt:
            template = custom_prompt
        else:
            template = """
            你是一位数据分析专家，请分析以下两个数据集之间的差异，并提供见解和可能的原因。

            差异信息:
            {diff_text}

            {context_info}

            请提供以下分析:
            1. 差异的总体概述
            2. 最显著的差异及其可能的业务影响
            3. 可能导致这些差异的原因
            4. 建议的后续步骤或调查方向

            分析:
            """
        
        context_info = ""
        if context:
            context_info = "额外上下文信息:\n"
            for key, value in context.items():
                context_info += f"- {key}: {value}\n"
        
        # 替换模板中的变量
        prompt = template.replace("{diff_text}", diff_text).replace("{context_info}", context_info)
        
        # 添加重试逻辑
        for attempt in range(self.retry_count):
            try:
                return self._call_llm_api(prompt)
            except Exception as e:
                error_msg = str(e)
                print(f"LLM分析失败 (尝试 {attempt+1}/{self.retry_count}): {error_msg}")
                
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"网络问题，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"连接到LLM服务失败: {error_msg}。请检查网络连接和API设置。")
                else:
                    # 其他类型的错误，直接抛出
                    raise
        
        raise RuntimeError("所有重试尝试均失败")
    
    def generate_sql_query(self, description: str, schema_info: Dict[str, Any] = None) -> str:
        """
        根据自然语言描述生成SQL查询
        
        Args:
            description: 自然语言描述的查询需求
            schema_info: 数据库表结构信息，如果为None则使用默认表结构
            
        Returns:
            生成的SQL查询语句
        """
        # 确保连接可用
        if not self._ensure_connection():
            raise ConnectionError("无法连接到LLM服务。请检查网络连接、API密钥和API基础URL是否正确。")
        
        # 如果没有提供表结构信息，使用默认的表结构描述
        if schema_info is None:
            schema_info_str = "请根据查询需求自行推断表结构"
        else:
            # 将表结构信息转换为字符串
            schema_info_str = ""
            for table_name, table_info in schema_info.items():
                schema_info_str += f"表名: {table_name}\n"
                schema_info_str += f"字段: {', '.join(table_info.get('columns', []))}\n"
                if 'primary_key' in table_info:
                    schema_info_str += f"主键: {table_info['primary_key']}\n"
                if 'foreign_keys' in table_info:
                    for fk in table_info['foreign_keys']:
                        schema_info_str += f"外键: {fk['column']} 引用 {fk['references_table']}.{fk['references_column']}\n"
                schema_info_str += "\n"
        
        # 创建提示模板
        template = """
        你是一位SQL专家，请根据以下数据库表结构信息和查询需求，生成有效的SQL查询语句。

        数据库表结构信息:
        {schema_info}

        查询需求:
        {description}

        请生成一个有效的SQL查询语句，确保语法正确，并且能够满足查询需求。
        只需要返回SQL语句，不需要任何解释。
        """
        
        # 替换模板中的变量
        prompt = template.replace("{schema_info}", schema_info_str).replace("{description}", description)
        
        # 添加重试逻辑
        for attempt in range(self.retry_count):
            try:
                sql = self._call_llm_api(prompt, system_message="你是一位SQL专家，能够根据表结构和需求生成准确的SQL查询语句。")
                # 清理生成的SQL（去除可能的引号和多余空格）
                sql = sql.strip().strip('`').strip()
                return sql
            except Exception as e:
                error_msg = str(e)
                print(f"SQL生成失败 (尝试 {attempt+1}/{self.retry_count}): {error_msg}")
                
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"网络问题，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"连接到LLM服务失败: {error_msg}。请检查网络连接和API设置。")
                else:
                    # 其他类型的错误，直接抛出
                    raise
        
        raise RuntimeError("所有重试尝试均失败")
    
    def analyze_query_results(self, query_results: Any, context: Dict[str, Any] = None) -> str:
        """
        分析查询结果
        
        Args:
            query_results: 查询结果，可以是DataFrame或其他格式
            context: 提供给LLM的额外上下文信息
            
        Returns:
            LLM生成的分析结果
        """
        # 确保连接可用
        if not self._ensure_connection():
            raise ConnectionError("无法连接到LLM服务。请检查网络连接、API密钥和API基础URL是否正确。")
        
        # 将查询结果转换为字符串
        if hasattr(query_results, 'to_string'):
            # 如果是DataFrame，使用to_string方法
            results_str = query_results.to_string()
        elif hasattr(query_results, 'to_dict'):
            # 如果有to_dict方法，先转换为字典再转为字符串
            import json
            results_str = json.dumps(query_results.to_dict(), ensure_ascii=False, indent=2)
        else:
            # 否则直接转换为字符串
            results_str = str(query_results)
        
        # 创建提示模板
        template = """
        你是一位数据分析专家，请分析以下查询结果，并提供见解和可能的业务含义。

        查询结果:
        {results}

        {context_info}

        请提供以下分析:
        1. 结果的总体概述
        2. 关键数据点及其含义
        3. 可能的业务洞察
        4. 建议的后续分析方向

        分析:
        """
        
        context_info = ""
        if context:
            context_info = "额外上下文信息:\n"
            for key, value in context.items():
                context_info += f"- {key}: {value}\n"
        
        # 替换模板中的变量
        prompt = template.replace("{results}", results_str).replace("{context_info}", context_info)
        
        # 添加重试逻辑
        for attempt in range(self.retry_count):
            try:
                return self._call_llm_api(prompt)
            except Exception as e:
                error_msg = str(e)
                print(f"结果分析失败 (尝试 {attempt+1}/{self.retry_count}): {error_msg}")
                
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"网络问题，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"连接到LLM服务失败: {error_msg}。请检查网络连接和API设置。")
                else:
                    # 其他类型的错误，直接抛出
                    raise
        
        raise RuntimeError("所有重试尝试均失败")
    
    def compare_datasets(self, dataset1: Any, dataset2: Any, 
                         dataset1_name: str = "数据集1", dataset2_name: str = "数据集2",
                         context: Dict[str, Any] = None) -> str:
        """
        比较两个数据集并分析差异
        
        Args:
            dataset1: 第一个数据集
            dataset2: 第二个数据集
            dataset1_name: 第一个数据集的名称
            dataset2_name: 第二个数据集的名称
            context: 提供给LLM的额外上下文信息
            
        Returns:
            LLM生成的差异分析结果
        """
        # 确保连接可用
        if not self._ensure_connection():
            raise ConnectionError("无法连接到LLM服务。请检查网络连接、API密钥和API基础URL是否正确。")
        
        # 将数据集转换为字符串
        if hasattr(dataset1, 'to_string'):
            # 如果是DataFrame，使用to_string方法
            dataset1_str = dataset1.to_string()
        else:
            # 否则直接转换为字符串
            dataset1_str = str(dataset1)
            
        if hasattr(dataset2, 'to_string'):
            # 如果是DataFrame，使用to_string方法
            dataset2_str = dataset2.to_string()
        else:
            # 否则直接转换为字符串
            dataset2_str = str(dataset2)
        
        # 创建提示模板
        template = """
        你是一位数据分析专家，请比较以下两个数据集，并分析它们之间的差异。

        {dataset1_name}:
        {dataset1}

        {dataset2_name}:
        {dataset2}

        {context_info}

        请提供以下分析:
        1. 两个数据集的总体差异概述
        2. 最显著的差异及其可能的业务影响
        3. 可能导致这些差异的原因
        4. 建议的后续步骤或调查方向

        分析:
        """
        
        context_info = ""
        if context:
            context_info = "额外上下文信息:\n"
            for key, value in context.items():
                context_info += f"- {key}: {value}\n"
        
        # 替换模板中的变量
        prompt = template.replace("{dataset1_name}", dataset1_name).replace("{dataset1}", dataset1_str) \
                        .replace("{dataset2_name}", dataset2_name).replace("{dataset2}", dataset2_str) \
                        .replace("{context_info}", context_info)
        
        # 添加重试逻辑
        for attempt in range(self.retry_count):
            try:
                return self._call_llm_api(prompt)
            except Exception as e:
                error_msg = str(e)
                print(f"数据集比较失败 (尝试 {attempt+1}/{self.retry_count}): {error_msg}")
                
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"网络问题，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"连接到LLM服务失败: {error_msg}。请检查网络连接和API设置。")
                else:
                    # 其他类型的错误，直接抛出
                    raise
        
        raise RuntimeError("所有重试尝试均失败")
    
    def generate_visualization_code(self, data_description: str, data_sample: Any = None) -> str:
        """
        生成可视化代码
        
        Args:
            data_description: 数据描述
            data_sample: 数据样本，可选
            
        Returns:
            生成的Python可视化代码
        """
        # 确保连接可用
        if not self._ensure_connection():
            raise ConnectionError("无法连接到LLM服务。请检查网络连接、API密钥和API基础URL是否正确。")
        
        # 将数据样本转换为字符串
        if data_sample is not None:
            if hasattr(data_sample, 'to_string'):
                # 如果是DataFrame，使用to_string方法
                data_sample_str = data_sample.head(10).to_string()
            else:
                # 否则直接转换为字符串
                data_sample_str = str(data_sample)[:1000]  # 限制长度
        else:
            data_sample_str = "未提供数据样本"
        
        # 创建提示模板
        template = """
        你是一位数据可视化专家，请根据以下数据描述和样本，生成Python代码来创建合适的可视化图表。

        数据描述:
        {data_description}

        数据样本:
        {data_sample}

        请生成使用matplotlib、seaborn或plotly的Python代码，创建能够有效展示数据特点的可视化图表。
        代码应该是完整的、可执行的，并包含必要的导入语句。
        请确保代码风格清晰，并添加适当的注释。

        Python代码:
        ```python
        """
        
        # 替换模板中的变量
        prompt = template.replace("{data_description}", data_description).replace("{data_sample}", data_sample_str)
        
        # 添加重试逻辑
        for attempt in range(self.retry_count):
            try:
                result = self._call_llm_api(prompt, system_message="你是一位数据可视化专家，能够生成高质量的Python可视化代码。")
                
                # 清理结果，提取代码部分
                if "```" in result:
                    import re
                    code_match = re.search(r'```(?:python)?(.*?)```', result, re.DOTALL)
                    if code_match:
                        return code_match.group(1).strip()
                
                return result
            except Exception as e:
                error_msg = str(e)
                print(f"可视化代码生成失败 (尝试 {attempt+1}/{self.retry_count}): {error_msg}")
                
                if "Connection" in error_msg or "timeout" in error_msg.lower():
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"网络问题，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"连接到LLM服务失败: {error_msg}。请检查网络连接和API设置。")
                else:
                    # 其他类型的错误，直接抛出
                    raise
        
        raise RuntimeError("所有重试尝试均失败")
    
    def update_config(self, **kwargs) -> None:
        """
        更新LLM配置
        
        Args:
            **kwargs: 要更新的配置参数，如model_name, temperature, max_tokens, api_base等
        """
        # 更新配置字典
        self.config.update(kwargs)
        
        # 更新实例变量
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
            
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
            
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
            
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        # 测试连接
        try:
            if self._test_connection():
                print(f"LLM配置已更新并成功连接: {self.config}")
            else:
                print(f"LLM配置已更新但连接测试失败: {self.config}")
        except Exception as e:
            print(f"LLM配置更新失败: {str(e)}")
            print("将在下次调用时重试连接")
            self.llm = None
    
    def analyze_diff_results(self, diff_results: Dict[str, Any], df1_name: str = "Dataset 1", 
                            df2_name: str = "Dataset 2", context: Dict[str, Any] = None) -> str:
        """
        直接分析DataDiffer生成的差异结果字典
        
        Args:
            diff_results: DataDiffer.compare_dataframes方法返回的差异结果字典
            df1_name: 第一个数据集的名称
            df2_name: 第二个数据集的名称
            context: 提供给LLM的额外上下文信息
            
        Returns:
            LLM生成的分析结果
        """
        from data_diff import DataDiffer
        
        # 将差异结果格式化为文本
        diff_text = DataDiffer.format_diff_results(diff_results, df1_name, df2_name)
        
        # 使用现有的analyze_differences方法分析格式化后的差异
        return self.analyze_differences(diff_text, context)
    
    def analyze_dataframes(self, df1: 'pd.DataFrame', df2: 'pd.DataFrame', 
                          key_columns: List[str] = None, df1_name: str = "Dataset 1", 
                          df2_name: str = "Dataset 2", context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        一站式分析两个DataFrame的差异
        
        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            key_columns: 用于匹配行的键列名列表，如果为None，则按索引比较
            df1_name: 第一个数据集的名称
            df2_name: 第二个数据集的名称
            context: 提供给LLM的额外上下文信息
            
        Returns:
            包含差异统计和分析结果的字典
        """
        from data_diff import DataDiffer
        
        # 使用DataDiffer比较两个DataFrame
        diff_results = DataDiffer.compare_dataframes(df1, df2, key_columns)
        
        # 格式化差异结果
        diff_text = DataDiffer.format_diff_results(diff_results, df1_name, df2_name)
        
        # 分析差异
        analysis = self.analyze_differences(diff_text, context)
        
        # 返回结果
        return {
            "diff_results": diff_results,
            "diff_text": diff_text,
            "analysis": analysis
        }


if __name__ == '__main__':
    analyzer = LLMAnalyzer()
    analyzer._test_connection()