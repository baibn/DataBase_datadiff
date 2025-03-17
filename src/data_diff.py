import pandas as pd
from deepdiff import DeepDiff
from typing import Dict, Any, List, Tuple, Union

class DataDiffer:
    """使用DeepDiff比较数据集之间的差异"""
    
    @staticmethod
    def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                          key_columns: List[str] = None) -> Dict[str, Any]:
        """
        比较两个DataFrame之间的差异
        
        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            key_columns: 用于匹配行的键列名列表，如果为None，则按索引比较
            
        Returns:
            包含差异信息的字典
        """
        if key_columns:
            # 如果提供了键列，则按键列进行比较
            df1_indexed = df1.set_index(key_columns)
            df2_indexed = df2.set_index(key_columns)
            
            # 找出共有的、仅在df1中的和仅在df2中的键
            common_keys = set(df1_indexed.index) & set(df2_indexed.index)
            only_in_df1 = set(df1_indexed.index) - set(df2_indexed.index)
            only_in_df2 = set(df2_indexed.index) - set(df1_indexed.index)
            
            # 比较共有键的行
            differences = {}
            for key in common_keys:
                row_diff = DeepDiff(df1_indexed.loc[key].to_dict(), 
                                   df2_indexed.loc[key].to_dict(),
                                   ignore_order=True)
                if row_diff:
                    differences[str(key)] = row_diff
            
            return {
                'common_differences': differences,
                'only_in_df1': list(only_in_df1),
                'only_in_df2': list(only_in_df2)
            }
        else:
            # 如果没有提供键列，则直接比较整个DataFrame
            return DeepDiff(df1.to_dict('records'), df2.to_dict('records'), ignore_order=True)
    
    @staticmethod
    def format_diff_results(diff_results: Dict[str, Any], df1_name: str = "Dataset 1", 
                           df2_name: str = "Dataset 2") -> str:
        """
        将差异结果格式化为可读的文本
        
        Args:
            diff_results: compare_dataframes方法返回的差异结果
            df1_name: 第一个数据集的名称
            df2_name: 第二个数据集的名称
            
        Returns:
            格式化后的差异描述
        """
        formatted_output = []
        
        if 'common_differences' in diff_results:
            # 处理按键比较的结果
            if diff_results['only_in_df1']:
                formatted_output.append(f"仅在 {df1_name} 中存在的记录: {len(diff_results['only_in_df1'])}")
                for key in diff_results['only_in_df1'][:10]:  # 限制输出数量
                    formatted_output.append(f"  - {key}")
                if len(diff_results['only_in_df1']) > 10:
                    formatted_output.append(f"    ... 以及 {len(diff_results['only_in_df1']) - 10} 条其他记录")
            
            if diff_results['only_in_df2']:
                formatted_output.append(f"仅在 {df2_name} 中存在的记录: {len(diff_results['only_in_df2'])}")
                for key in diff_results['only_in_df2'][:10]:  # 限制输出数量
                    formatted_output.append(f"  - {key}")
                if len(diff_results['only_in_df2']) > 10:
                    formatted_output.append(f"    ... 以及 {len(diff_results['only_in_df2']) - 10} 条其他记录")
            
            if diff_results['common_differences']:
                formatted_output.append(f"共有记录中的差异: {len(diff_results['common_differences'])}")
                for key, diff in list(diff_results['common_differences'].items())[:10]:
                    formatted_output.append(f"  - 记录 {key}:")
                    
                    if 'values_changed' in diff:
                        for change_path, change in diff['values_changed'].items():
                            field = change_path.replace("root['", "").replace("']", "")
                            formatted_output.append(f"    - 字段 '{field}' 从 '{change['old_value']}' 变为 '{change['new_value']}'")
                    
                    if 'type_changes' in diff:
                        for change_path, change in diff['type_changes'].items():
                            field = change_path.replace("root['", "").replace("']", "")
                            formatted_output.append(f"    - 字段 '{field}' 类型从 {type(change['old_value']).__name__} 变为 {type(change['new_value']).__name__}")
                
                if len(diff_results['common_differences']) > 10:
                    formatted_output.append(f"    ... 以及 {len(diff_results['common_differences']) - 10} 条其他差异记录")
        else:
            # 处理直接比较的结果
            for diff_type, diff_items in diff_results.items():
                if diff_type == 'dictionary_item_added':
                    formatted_output.append(f"在 {df2_name} 中新增的项: {len(diff_items)}")
                elif diff_type == 'dictionary_item_removed':
                    formatted_output.append(f"在 {df1_name} 中存在但在 {df2_name} 中不存在的项: {len(diff_items)}")
                elif diff_type == 'values_changed':
                    formatted_output.append(f"值发生变化的项: {len(diff_items)}")
                    for change_path, change in list(diff_items.items())[:10]:
                        formatted_output.append(f"  - {change_path}: 从 '{change['old_value']}' 变为 '{change['new_value']}'")
                    if len(diff_items) > 10:
                        formatted_output.append(f"    ... 以及 {len(diff_items) - 10} 条其他变化")
        
        return "\n".join(formatted_output)