"""
File upload utility for handling document uploads in the application.
Supports PDF, DOCX, PPTX, TXT, and other document formats.
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib
from datetime import datetime


class FileUploadManager:
    """Manages file uploads with validation and storage"""
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.pptx', '.ppt',
        '.txt', '.md', '.csv', '.xlsx', '.xls'
    }
    
    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self, upload_dir: str = "./data"):
        """
        Initialize file upload manager
        
        Args:
            upload_dir: Directory where uploaded files will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file extension
        if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            return False, f"File type {file_path.suffix} not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({self.MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"
        
        # Check if file is empty
        if file_size == 0:
            return False, "File is empty"
        
        return True, ""
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Generate SHA256 hash of file content
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_uploaded_file(self, source_path: str, custom_name: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Save uploaded file to upload directory
        
        Args:
            source_path: Path to the source file
            custom_name: Optional custom name for the file
            
        Returns:
            Tuple of (success, saved_path, message)
        """
        # Validate file
        is_valid, error_msg = self.validate_file(source_path)
        if not is_valid:
            return False, "", error_msg
        
        source_path = Path(source_path)
        
        # Generate destination filename
        if custom_name:
            # Sanitize custom name
            custom_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_', '.'))
            dest_filename = custom_name
        else:
            dest_filename = source_path.name
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = dest_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            dest_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            dest_filename = f"{dest_filename}_{timestamp}"
        
        dest_path = self.upload_dir / dest_filename
        
        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            file_hash = self.get_file_hash(str(dest_path))
            return True, str(dest_path), f"File uploaded successfully: {dest_filename} (Hash: {file_hash[:8]}...)"
        except Exception as e:
            return False, "", f"Error saving file: {str(e)}"
    
    def save_multiple_files(self, file_paths: List[str]) -> List[Tuple[str, bool, str]]:
        """
        Save multiple uploaded files
        
        Args:
            file_paths: List of paths to files to upload
            
        Returns:
            List of tuples (filename, success, message)
        """
        results = []
        for file_path in file_paths:
            success, saved_path, message = self.save_uploaded_file(file_path)
            filename = Path(file_path).name
            results.append((filename, success, message))
        return results
    
    def list_uploaded_files(self) -> List[dict]:
        """
        List all files in upload directory
        
        Returns:
            List of file information dictionaries
        """
        files = []
        for file_path in self.upload_dir.glob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.ALLOWED_EXTENSIONS:
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'extension': file_path.suffix
                })
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def delete_file(self, filename: str) -> Tuple[bool, str]:
        """
        Delete a file from upload directory
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            Tuple of (success, message)
        """
        file_path = self.upload_dir / filename
        
        if not file_path.exists():
            return False, "File not found"
        
        if not file_path.is_file():
            return False, "Not a file"
        
        try:
            file_path.unlink()
            return True, f"File {filename} deleted successfully"
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"
    
    def get_upload_stats(self) -> dict:
        """
        Get statistics about uploaded files
        
        Returns:
            Dictionary with upload statistics
        """
        files = self.list_uploaded_files()
        total_size = sum(f['size'] for f in files)
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'file_types': {ext: len([f for f in files if f['extension'] == ext]) 
                          for ext in set(f['extension'] for f in files)},
            'upload_dir': str(self.upload_dir)
        }
