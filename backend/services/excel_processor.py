"""
Excel Processing Service
Extracts data from Excel files (.xlsx, .xls)
"""

import pandas as pd
import io
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Service for processing Excel files and extracting data"""
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls', '.xlsm']
    
    def extract_data_from_excel(self, excel_bytes: bytes, filename: str = None) -> Dict:
        """
        Extract data from Excel file
        
        Args:
            excel_bytes: Excel file content as bytes
            filename: Optional filename for error tracking
            
        Returns:
            Dictionary with extracted data chunks and metadata
        """
        try:
            print(f"[Excel] extract_data_from_excel: filename={filename}, bytes_len={len(excel_bytes) if excel_bytes else 0}")
            
            # Read Excel file
            excel_file = io.BytesIO(excel_bytes)
            
            # Try to read all sheets
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
            except Exception as e:
                print(f"[Excel] openpyxl failed, trying xlrd: {e}")
                try:
                    excel_file.seek(0)
                    excel_data = pd.read_excel(excel_file, sheet_name=None, engine='xlrd')
                except Exception as e2:
                    print(f"[Excel] xlrd also failed: {e2}")
                    raise Exception(f"Could not read Excel file. Error: {str(e2)}")
            
            if not excel_data:
                raise Exception("No data found in Excel file")
            
            chunks = []
            total_rows = 0
            sheet_names = list(excel_data.keys())
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                
                # Convert DataFrame to string representation with context
                sheet_info = f"Sheet: {sheet_name}\n"
                sheet_info += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
                
                # Add column headers
                sheet_info += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
                
                # Process data in chunks (rows)
                for idx, row in df.iterrows():
                    # Create a readable text representation of the row
                    row_text = f"Row {idx + 1}:\n"
                    
                    # Add each column value
                    for col in df.columns:
                        value = row[col]
                        # Handle NaN values
                        if pd.isna(value):
                            value = "N/A"
                        else:
                            value = str(value)
                        row_text += f"  {col}: {value}\n"
                    
                    chunks.append({
                        'text': sheet_info + row_text,
                        'metadata': {
                            'sheet_name': sheet_name,
                            'row_index': int(idx),
                            'chunk_type': 'row',
                            'filename': filename
                        }
                    })
                    total_rows += 1
                
                # Also create a summary chunk for the sheet
                summary_text = f"Sheet Summary: {sheet_name}\n"
                summary_text += f"Total Rows: {len(df)}\n"
                summary_text += f"Total Columns: {len(df.columns)}\n"
                summary_text += f"Column Names: {', '.join(df.columns.astype(str))}\n"
                
                # Add sample data (first few rows)
                if len(df) > 0:
                    summary_text += "\nSample Data (first 3 rows):\n"
                    for idx in range(min(3, len(df))):
                        summary_text += f"Row {idx + 1}: "
                        row_values = []
                        for col in df.columns:
                            value = df.iloc[idx][col]
                            if pd.isna(value):
                                value = "N/A"
                            else:
                                value = str(value)
                            row_values.append(f"{col}={value}")
                        summary_text += ", ".join(row_values) + "\n"
                
                chunks.append({
                    'text': summary_text,
                    'metadata': {
                        'sheet_name': sheet_name,
                        'chunk_type': 'summary',
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'filename': filename
                    }
                })
            
            result = {
                'chunks': chunks,
                'total_chunks': len(chunks),
                'metadata': {
                    'filename': filename,
                    'total_sheets': len(sheet_names),
                    'sheet_names': sheet_names,
                    'total_rows': total_rows,
                    'file_type': 'excel'
                }
            }
            
            print(f"[Excel] Extraction successful: {len(chunks)} chunks, {total_rows} rows, {len(sheet_names)} sheets")
            return result
            
        except Exception as e:
            error_msg = f"Error extracting data from Excel file: {str(e)}"
            print(f"[Excel] {error_msg}")
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    def is_supported(self, filename: str) -> bool:
        """Check if file format is supported"""
        if not filename:
            return False
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)

