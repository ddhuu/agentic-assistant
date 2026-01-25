import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from langchain.schema import AIMessage
from utils.logger import log
from config.llm import gemini

class DataAnalysisAgent:
    """Agent chuyên phân tích và so sánh dữ liệu từ nhiều nguồn."""
    
    def __init__(self, model=None):
        self.name = "data_analysis"
        self.model = model or gemini
        self.session_id = None
    
    def extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """
        Trích xuất bảng từ văn bản.
        
        Args:
            text: Nội dung văn bản có thể chứa bảng
            
        Returns:
            List các DataFrame đã trích xuất
        """
        tables = []
        
        # Tìm các dòng có dấu hiệu của bảng (có nhiều dấu | hoặc nhiều khoảng trắng)
        lines = text.split('\n')
        table_start_indices = []
        in_table = False
        current_table_lines = []
        
        for i, line in enumerate(lines):
            # Kiểm tra xem dòng có phải là một phần của bảng không
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    table_start_indices.append(i)
                current_table_lines.append(line)
            elif in_table and (line.strip() == '' or i == len(lines) - 1):
                # Kết thúc bảng
                in_table = False
                if current_table_lines:
                    try:
                        # Cố gắng chuyển đổi thành DataFrame
                        df = self._convert_text_to_dataframe(current_table_lines)
                        if not df.empty:
                            tables.append(df)
                    except Exception as e:
                        log(f"Lỗi khi chuyển đổi bảng: {e}", level='error')
                current_table_lines = []
            elif in_table:
                current_table_lines.append(line)
        
        # Xử lý bảng cuối cùng nếu có
        if in_table and current_table_lines:
            try:
                df = self._convert_text_to_dataframe(current_table_lines)
                if not df.empty:
                    tables.append(df)
            except Exception as e:
                log(f"Lỗi khi chuyển đổi bảng cuối cùng: {e}", level='error')
        
        return tables
    
    def _convert_text_to_dataframe(self, table_lines: List[str]) -> pd.DataFrame:
        """
        Chuyển đổi các dòng văn bản thành DataFrame.
        
        Args:
            table_lines: Danh sách các dòng văn bản đại diện cho bảng
            
        Returns:
            DataFrame đã chuyển đổi
        """
        # Xử lý bảng dạng markdown
        if all('|' in line for line in table_lines):
            # Loại bỏ các ký tự | ở đầu và cuối dòng
            cleaned_lines = [line.strip('|').strip() for line in table_lines]
            
            # Loại bỏ dòng phân cách (dòng chỉ chứa '-' và '|')
            cleaned_lines = [line for line in cleaned_lines if not re.match(r'^[\s\-\|]+$', line)]
            
            if not cleaned_lines:
                return pd.DataFrame()
            
            # Tách các cột
            rows = [re.split(r'\s*\|\s*', line) for line in cleaned_lines]
            
            # Lấy header từ dòng đầu tiên
            header = rows[0]
            data = rows[1:]
            
            # Tạo DataFrame
            df = pd.DataFrame(data, columns=header)
            return df
        
        # Xử lý bảng dạng khoảng trắng
        else:
            # Tách các cột dựa trên khoảng trắng
            rows = [re.split(r'\s{2,}', line.strip()) for line in table_lines]
            
            # Kiểm tra xem tất cả các hàng có cùng số cột không
            if not all(len(row) == len(rows[0]) for row in rows):
                # Nếu không, thử phương pháp khác
                return pd.DataFrame()
            
            # Lấy header từ dòng đầu tiên
            header = rows[0]
            data = rows[1:]
            
            # Tạo DataFrame
            df = pd.DataFrame(data, columns=header)
            return df
    
    def extract_numeric_data(self, text: str, keywords: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Trích xuất dữ liệu số từ văn bản dựa trên từ khóa.
        
        Args:
            text: Nội dung văn bản
            keywords: Danh sách từ khóa cần tìm (ví dụ: "doanh thu", "lợi nhuận")
            
        Returns:
            Dictionary chứa dữ liệu đã trích xuất
        """
        results = {}
        
        # Tìm các đoạn văn bản có chứa từ khóa
        for keyword in keywords:
            keyword_lower = keyword.lower()
            results[keyword] = {}
            
            # Tìm các năm (2020, 2021, 2022, 2023, 2024, etc.)
            years = re.findall(r'20\d{2}', text)
            unique_years = sorted(set(years))
            
            for year in unique_years:
                # Tìm các số liệu gần từ khóa và năm
                pattern = fr'(?i)({keyword_lower}[^.]*?{year}|{year}[^.]*?{keyword_lower})[^.]*?(\d+[\.,]?\d*)\s*(tỷ|triệu|nghìn|tr|ty)?'
                matches = re.findall(pattern, text)
                
                if matches:
                    for match in matches:
                        value_str = match[1].replace(',', '.')
                        try:
                            value = float(value_str)
                            # Áp dụng đơn vị
                            if match[2].lower() in ['tỷ', 'ty']:
                                value *= 1_000_000_000
                            elif match[2].lower() in ['triệu', 'tr']:
                                value *= 1_000_000
                            elif match[2].lower() == 'nghìn':
                                value *= 1_000
                                
                            results[keyword][year] = value
                            break
                        except ValueError:
                            continue
        
        return results
    
    def compare_data(self, data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        So sánh dữ liệu giữa các năm.
        
        Args:
            data: Dictionary chứa dữ liệu đã trích xuất
            
        Returns:
            Dictionary chứa kết quả so sánh
        """
        comparison = {}
        
        for metric, year_data in data.items():
            comparison[metric] = {}
            years = sorted(year_data.keys())
            
            if len(years) < 2:
                comparison[metric]["status"] = "Không đủ dữ liệu để so sánh"
                continue
                
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                
                if prev_year in year_data and curr_year in year_data:
                    prev_value = year_data[prev_year]
                    curr_value = year_data[curr_year]
                    
                    absolute_change = curr_value - prev_value
                    percent_change = (absolute_change / prev_value) * 100 if prev_value != 0 else float('inf')
                    
                    comparison[metric][f"{prev_year}-{curr_year}"] = {
                        "absolute_change": absolute_change,
                        "percent_change": percent_change,
                        "status": "Tăng" if absolute_change > 0 else "Giảm" if absolute_change < 0 else "Không đổi"
                    }
        
        return comparison
    
    def format_numeric_value(self, value: float) -> str:
        """
        Định dạng giá trị số thành chuỗi dễ đọc.
        
        Args:
            value: Giá trị số cần định dạng
            
        Returns:
            Chuỗi đã định dạng
        """
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} tỷ"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f} triệu"
        elif value >= 1_000:
            return f"{value / 1_000:.2f} nghìn"
        else:
            return f"{value:.2f}"
    
    def generate_comparison_report(self, data: Dict[str, Dict[str, float]], comparison: Dict[str, Dict[str, Any]]) -> str:
        """
        Tạo báo cáo so sánh từ dữ liệu đã phân tích.
        
        Args:
            data: Dictionary chứa dữ liệu đã trích xuất
            comparison: Dictionary chứa kết quả so sánh
            
        Returns:
            Báo cáo so sánh dạng văn bản
        """
        report = "# BÁO CÁO SO SÁNH DỮ LIỆU\n\n"
        
        for metric, metric_data in data.items():
            report += f"## {metric.upper()}\n\n"
            
            # Hiển thị dữ liệu theo năm
            report += "### Dữ liệu theo năm\n\n"
            years = sorted(metric_data.keys())
            for year in years:
                report += f"- Năm {year}: {self.format_numeric_value(metric_data[year])}\n"
            
            report += "\n"
            
            # Hiển thị so sánh
            if metric in comparison:
                report += "### So sánh giữa các năm\n\n"
                for period, period_data in comparison[metric].items():
                    if isinstance(period_data, dict) and "status" in period_data:
                        if period_data["status"] in ["Tăng", "Giảm", "Không đổi"]:
                            report += f"- {period}: {period_data['status']} "
                            report += f"{self.format_numeric_value(abs(period_data['absolute_change']))} "
                            report += f"({period_data['percent_change']:.2f}%)\n"
                        else:
                            report += f"- {period}: {period_data['status']}\n"
            
            report += "\n"
        
        return report
    
    async def analyze_contents(self, contents: Dict[str, str], metrics: List[str]) -> Dict[str, Any]:
        """
        Phân tích nội dung từ nhiều nguồn và trích xuất thông tin liên quan.
        
        Args:
            contents: Dictionary chứa nội dung từ các file, với key là tên file
            metrics: Danh sách các chỉ số cần phân tích (ví dụ: "doanh thu", "lợi nhuận")
            
        Returns:
            Dictionary chứa kết quả phân tích
        """
        results = {
            "extracted_data": {},
            "tables": {},
            "comparison": {},
            "report": ""
        }
        
        all_data = {}
        
        # Phân tích từng file
        for file_name, content in contents.items():
            log(f"Đang phân tích file: {file_name}")
            
            # Trích xuất bảng
            tables = self.extract_tables_from_text(content)
            if tables:
                results["tables"][file_name] = tables
            
            # Trích xuất dữ liệu số
            file_data = self.extract_numeric_data(content, metrics)
            results["extracted_data"][file_name] = file_data
            
            # Gộp dữ liệu từ tất cả các file
            for metric, year_data in file_data.items():
                if metric not in all_data:
                    all_data[metric] = {}
                all_data[metric].update(year_data)
        
        # So sánh dữ liệu
        comparison = self.compare_data(all_data)
        results["comparison"] = comparison
        
        # Tạo báo cáo
        report = self.generate_comparison_report(all_data, comparison)
        results["report"] = report
        
        return results
    
    async def invoke(self, query: str, session_id: str = None) -> Union[str, Dict[str, Any]]:
        """
        Xử lý yêu cầu phân tích dữ liệu.
        
        Args:
            query: Yêu cầu từ người dùng
            session_id: ID phiên làm việc
            
        Returns:
            Kết quả phân tích hoặc thông báo lỗi
        """
        self.session_id = session_id
        
        try:
            # Phân tích yêu cầu để xác định các chỉ số cần phân tích
            analysis_prompt = f"""
            Dựa vào yêu cầu sau đây, hãy xác định các chỉ số tài chính hoặc kinh doanh cần phân tích:
            
            "{query}"
            
            Chỉ trả về danh sách các chỉ số, mỗi chỉ số trên một dòng, không có giải thích.
            Ví dụ:
            doanh thu
            lợi nhuận
            chi phí
            """
            
            response = await self.model.ainvoke(analysis_prompt)
            metrics = [line.strip().lower() for line in response.content.strip().split('\n') if line.strip()]
            
            if not metrics:
                metrics = ["doanh thu", "lợi nhuận"]
            
            log(f"Các chỉ số cần phân tích: {metrics}")
            
            return {
                "metrics": metrics,
                "message": f"Đã xác định {len(metrics)} chỉ số cần phân tích: {', '.join(metrics)}"
            }
            
        except Exception as e:
            log(f"Lỗi khi phân tích dữ liệu: {e}", level='error')
            return f"Xin lỗi, tôi gặp lỗi khi phân tích dữ liệu: {str(e)}"
