import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect


class DatabaseConnector:
    """数据库连接器，用于从不同的数据库中提取数据"""

    def __init__(self, connection_string: str, llm_api_key: Optional[str] = None, llm_api_url: Optional[str] = None):
        """
        初始化数据库连接器
        
        Args:
            connection_string: 数据库连接字符串，例如 'sqlite:///example.db' 或 'postgresql://user:password@localhost:5432/dbname'
            llm_api_key: 大模型API密钥，用于生成SQL查询
            llm_api_url: 大模型API地址
        """
        self.engine = create_engine(connection_string)
        self.llm_api_key = llm_api_key
        self.llm_api_url = llm_api_url or "https://api.siliconflow.cn/v1/chat/completions"
        self.model_name = "deepseek-ai/DeepSeek-V2.5"  # 默认使用DeepSeek-V2.5模型

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        执行SQL查询并返回结果
        
        Args:
            query: SQL查询语句
            
        Returns:
            查询结果的DataFrame
        """
        return pd.read_sql(query, self.engine)

    def get_table_data(self, table_name: str, columns: Optional[list] = None,
                       where_clause: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取表格数据
        
        Args:
            table_name: 表名
            columns: 要选择的列名列表，默认为所有列
            where_clause: WHERE子句，默认为None
            limit: 限制返回的行数，默认为None
            
        Returns:
            表格数据的DataFrame
        """
        cols_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols_str} FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        return self.execute_query(query)

    def get_table_structure(self, table_name: str) -> List[Dict[str, Any]]:
        """
        获取表结构信息
        
        Args:
            table_name: 表名
            
        Returns:
            包含列信息的字典列表
        """
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)

    def generate_query(self, table_name: str, query_type: str = 'SELECT',
                       conditions: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None) -> str:
        """
        根据表结构自动生成SQL查询语句
        
        Args:
            table_name: 表名
            query_type: 查询类型，如'SELECT'、'COUNT'等
            conditions: 查询条件，格式为{列名: 值}
            limit: 限制返回的行数
            
        Returns:
            生成的SQL查询语句
        """
        # 获取表结构
        columns = self.get_table_structure(table_name)
        column_names = [col['name'] for col in columns]

        # 根据查询类型生成基本查询
        if query_type.upper() == 'SELECT':
            query = f"SELECT {', '.join(column_names)} FROM {table_name}"
        elif query_type.upper() == 'COUNT':
            query = f"SELECT COUNT(*) FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        # 添加条件
        if conditions:
            where_clauses = []
            for col, value in conditions.items():
                if col in column_names:
                    # 根据列类型处理值
                    col_type = next((c['type'] for c in columns if c['name'] == col), None)
                    if hasattr(col_type, 'python_type') and col_type.python_type in (str, bytes):
                        where_clauses.append(f"{col} = '{value}'")
                    else:
                        where_clauses.append(f"{col} = {value}")

            if where_clauses:
                query += f" WHERE {' AND '.join(where_clauses)}"

        # 添加限制
        if limit:
            query += f" LIMIT {limit}"

        return query

    def get_sample_data(self, table_name: str, sample_size: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        获取表的样本数据和生成的查询语句
        
        Args:
            table_name: 表名
            sample_size: 样本大小
            
        Returns:
            (样本数据DataFrame, 生成的SQL查询语句)
        """
        query = self.generate_query(table_name, limit=sample_size)
        df = self.execute_query(query)
        return df, query

    def generate_query_with_llm(self, table_name: str, query_description: str,
                                sample_size: int = 5) -> str:
        """
        使用大模型根据表结构和查询描述生成SQL查询
        
        Args:
            table_name: 表名
            query_description: 用自然语言描述的查询需求
            sample_size: 获取样本数据的大小，用于帮助模型理解数据
            
        Returns:
            生成的SQL查询语句
        """
        if not self.llm_api_key:
            raise ValueError("未提供大模型API密钥，无法使用LLM生成查询")

        # 获取表结构
        columns = self.get_table_structure(table_name)
        column_info = "\n".join([f"- {col['name']}: {col['type']}" for col in columns])

        # 获取样本数据
        sample_df, _ = self.get_sample_data(table_name, sample_size)
        sample_data = sample_df.head().to_string()

        # 构建提示词
        prompt = f"""
        我需要为以下表生成一个SQL查询:
        
        表名: {table_name}
        
        表结构:
        {column_info}
        
        样本数据:
        {sample_data}
        
        请根据以下描述生成SQL查询:
        {query_description}
        
        只返回SQL查询语句，不要包含任何解释。
        """

        # 调用大模型API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个SQL专家，能够根据表结构和需求生成准确的SQL查询语句。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }

        try:
            response = requests.post(self.llm_api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            sql_query = result["choices"][0]["message"]["content"].strip()
            return sql_query
        except Exception as e:
            raise Exception(f"调用大模型API失败: {str(e)}")

    def execute_llm_query(self, table_name: str, query_description: str) -> Tuple[pd.DataFrame, str]:
        """
        使用大模型生成并执行SQL查询
        
        Args:
            table_name: 表名
            query_description: 用自然语言描述的查询需求
            
        Returns:
            (查询结果DataFrame, 生成的SQL查询语句)
        """
        sql_query = self.generate_query_with_llm(table_name, query_description)
        result_df = self.execute_query(sql_query)
        return result_df, sql_query



