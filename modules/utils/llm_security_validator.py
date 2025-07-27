"""
LLM-enhanced security and validation system for Cherry AI Streamlit Platform.
Uses LLM to analyze file content, detect suspicious patterns, and provide security recommendations.
"""

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pandas as pd
import numpy as np

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    MALICIOUS_CODE = "malicious_code"
    DATA_INJECTION = "data_injection"
    PRIVACY_VIOLATION = "privacy_violation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    OVERSIZED_DATA = "oversized_data"
    INVALID_FORMAT = "invalid_format"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

@dataclass
class SecurityThreat:
    """Security threat information"""
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    location: str
    recommendation: str
    confidence: float
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """File validation result"""
    is_safe: bool
    threats: List[SecurityThreat]
    file_info: Dict[str, Any]
    processing_time: float
    llm_analysis: Optional[Dict[str, Any]] = None

class FileSecurityAnalyzer:
    """Analyze files for security threats using LLM"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        
        # Security patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'__import__\s*\(',  # Python imports
            r'subprocess\.',  # Subprocess calls
            r'os\.system',  # OS system calls
            r'DROP\s+TABLE',  # SQL injection
            r'DELETE\s+FROM',  # SQL deletion
            r'INSERT\s+INTO.*VALUES',  # SQL insertion
        ]
        
        # Sensitive data patterns
        self.sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
        ]
        
        # File size limits (MB)
        self.size_limits = {
            'csv': 100,
            'json': 50,
            'xlsx': 200,
            'txt': 10,
            'default': 50
        }
    
    async def analyze_file(self, file_path: str, file_content: bytes = None, metadata: Dict[str, Any] = None) -> ValidationResult:
        """Analyze file for security threats"""
        start_time = time.time()
        threats = []
        
        try:
            # Get file info
            file_info = self._get_file_info(file_path, file_content)
            
            # Basic security checks
            threats.extend(await self._check_file_size(file_info))
            threats.extend(await self._check_file_type(file_info))
            threats.extend(await self._check_malicious_patterns(file_content or b''))
            
            # Content-specific analysis
            if file_content:
                threats.extend(await self._analyze_content_structure(file_content, file_info))
                threats.extend(await self._check_sensitive_data(file_content))
            
            # LLM-based analysis
            llm_analysis = None
            if self.llm_client and file_content:
                llm_analysis = await self._llm_security_analysis(file_content, file_info, metadata)
                if llm_analysis and llm_analysis.get('threats'):
                    threats.extend(self._parse_llm_threats(llm_analysis['threats']))
            
            # Determine overall safety
            is_safe = not any(threat.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] for threat in threats)
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                is_safe=is_safe,
                threats=threats,
                file_info=file_info,
                processing_time=processing_time,
                llm_analysis=llm_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            
            # Return unsafe result on error
            return ValidationResult(
                is_safe=False,
                threats=[SecurityThreat(
                    threat_type=ThreatType.INVALID_FORMAT,
                    severity=SecurityLevel.HIGH,
                    description=f"Analysis failed: {str(e)}",
                    location=file_path,
                    recommendation="Manual review required",
                    confidence=1.0
                )],
                file_info={'error': str(e)},
                processing_time=time.time() - start_time
            )
    
    def _get_file_info(self, file_path: str, file_content: bytes = None) -> Dict[str, Any]:
        """Get basic file information"""
        info = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1].lower(),
            'mime_type': mimetypes.guess_type(file_path)[0],
        }
        
        if file_content:
            info['size_bytes'] = len(file_content)
            info['size_mb'] = len(file_content) / (1024 * 1024)
            info['hash'] = hashlib.md5(file_content).hexdigest()
        elif os.path.exists(file_path):
            stat = os.stat(file_path)
            info['size_bytes'] = stat.st_size
            info['size_mb'] = stat.st_size / (1024 * 1024)
            info['modified'] = datetime.fromtimestamp(stat.st_mtime)
        
        return info
    
    async def _check_file_size(self, file_info: Dict[str, Any]) -> List[SecurityThreat]:
        """Check file size limits"""
        threats = []
        
        size_mb = file_info.get('size_mb', 0)
        extension = file_info.get('extension', '').lstrip('.')
        
        limit = self.size_limits.get(extension, self.size_limits['default'])
        
        if size_mb > limit:
            threats.append(SecurityThreat(
                threat_type=ThreatType.OVERSIZED_DATA,
                severity=SecurityLevel.MEDIUM if size_mb < limit * 2 else SecurityLevel.HIGH,
                description=f"File size {size_mb:.1f}MB exceeds limit of {limit}MB",
                location=file_info.get('name', 'unknown'),
                recommendation=f"Reduce file size or split into smaller chunks",
                confidence=1.0,
                metadata={'size_mb': size_mb, 'limit_mb': limit}
            ))
        
        return threats
    
    async def _check_file_type(self, file_info: Dict[str, Any]) -> List[SecurityThreat]:
        """Check file type validity"""
        threats = []
        
        extension = file_info.get('extension', '').lower()
        mime_type = file_info.get('mime_type')
        
        # Allowed file types
        allowed_extensions = {'.csv', '.json', '.xlsx', '.xls', '.txt', '.parquet'}
        allowed_mime_types = {
            'text/csv', 'application/json', 'text/plain',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        if extension not in allowed_extensions:
            threats.append(SecurityThreat(
                threat_type=ThreatType.INVALID_FORMAT,
                severity=SecurityLevel.HIGH,
                description=f"File type '{extension}' is not allowed",
                location=file_info.get('name', 'unknown'),
                recommendation="Use supported file formats: CSV, JSON, Excel, TXT",
                confidence=1.0,
                metadata={'extension': extension, 'mime_type': mime_type}
            ))
        
        return threats
    
    async def _check_malicious_patterns(self, file_content: bytes) -> List[SecurityThreat]:
        """Check for malicious code patterns"""
        threats = []
        
        try:
            # Try to decode as text
            text_content = file_content.decode('utf-8', errors='ignore')
            
            for pattern in self.malicious_patterns:
                matches = re.finditer(pattern, text_content, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.MALICIOUS_CODE,
                        severity=SecurityLevel.CRITICAL,
                        description=f"Potentially malicious pattern detected: {pattern}",
                        location=f"Position {match.start()}-{match.end()}",
                        recommendation="Remove malicious code before processing",
                        confidence=0.8,
                        metadata={'pattern': pattern, 'match': match.group()[:100]}
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error checking malicious patterns: {str(e)}")
        
        return threats
    
    async def _check_sensitive_data(self, file_content: bytes) -> List[SecurityThreat]:
        """Check for sensitive data patterns"""
        threats = []
        
        try:
            text_content = file_content.decode('utf-8', errors='ignore')
            
            for pattern in self.sensitive_patterns:
                matches = re.finditer(pattern, text_content)
                
                for match in matches:
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.PRIVACY_VIOLATION,
                        severity=SecurityLevel.MEDIUM,
                        description=f"Potentially sensitive data detected",
                        location=f"Position {match.start()}-{match.end()}",
                        recommendation="Review and anonymize sensitive data",
                        confidence=0.6,
                        metadata={'pattern_type': 'sensitive_data'}
                    ))
        
        except Exception as e:
            self.logger.warning(f"Error checking sensitive data: {str(e)}")
        
        return threats
    
    async def _analyze_content_structure(self, file_content: bytes, file_info: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze content structure for anomalies"""
        threats = []
        
        try:
            extension = file_info.get('extension', '').lower()
            
            if extension == '.csv':
                threats.extend(await self._analyze_csv_structure(file_content))
            elif extension == '.json':
                threats.extend(await self._analyze_json_structure(file_content))
            
        except Exception as e:
            self.logger.warning(f"Error analyzing content structure: {str(e)}")
        
        return threats
    
    async def _analyze_csv_structure(self, file_content: bytes) -> List[SecurityThreat]:
        """Analyze CSV structure for anomalies"""
        threats = []
        
        try:
            # Try to parse as CSV
            import io
            text_content = file_content.decode('utf-8', errors='ignore')
            
            # Check for extremely long lines (potential attack)
            lines = text_content.split('\n')
            for i, line in enumerate(lines[:100]):  # Check first 100 lines
                if len(line) > 10000:  # 10KB per line
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.SUSPICIOUS_PATTERN,
                        severity=SecurityLevel.MEDIUM,
                        description=f"Extremely long line detected (line {i+1})",
                        location=f"Line {i+1}",
                        recommendation="Review line content for potential issues",
                        confidence=0.7
                    ))
            
            # Check for unusual number of columns
            if lines:
                first_line_cols = len(lines[0].split(','))
                if first_line_cols > 1000:  # More than 1000 columns
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.SUSPICIOUS_PATTERN,
                        severity=SecurityLevel.MEDIUM,
                        description=f"Unusual number of columns: {first_line_cols}",
                        location="CSV header",
                        recommendation="Verify data structure is correct",
                        confidence=0.6
                    ))
        
        except Exception as e:
            threats.append(SecurityThreat(
                threat_type=ThreatType.INVALID_FORMAT,
                severity=SecurityLevel.MEDIUM,
                description=f"CSV parsing error: {str(e)}",
                location="File structure",
                recommendation="Verify CSV format is valid",
                confidence=0.8
            ))
        
        return threats
    
    async def _analyze_json_structure(self, file_content: bytes) -> List[SecurityThreat]:
        """Analyze JSON structure for anomalies"""
        threats = []
        
        try:
            text_content = file_content.decode('utf-8', errors='ignore')
            json_data = json.loads(text_content)
            
            # Check for deeply nested structures (potential DoS)
            max_depth = self._get_json_depth(json_data)
            if max_depth > 50:
                threats.append(SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_PATTERN,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Deeply nested JSON structure (depth: {max_depth})",
                    location="JSON structure",
                    recommendation="Flatten JSON structure to reduce complexity",
                    confidence=0.7
                ))
            
            # Check for extremely large arrays
            large_arrays = self._find_large_arrays(json_data)
            for path, size in large_arrays:
                if size > 100000:  # More than 100k items
                    threats.append(SecurityThreat(
                        threat_type=ThreatType.OVERSIZED_DATA,
                        severity=SecurityLevel.MEDIUM,
                        description=f"Large array with {size} items",
                        location=path,
                        recommendation="Consider splitting large arrays",
                        confidence=0.8
                    ))
        
        except json.JSONDecodeError as e:
            threats.append(SecurityThreat(
                threat_type=ThreatType.INVALID_FORMAT,
                severity=SecurityLevel.MEDIUM,
                description=f"Invalid JSON format: {str(e)}",
                location="JSON parsing",
                recommendation="Fix JSON syntax errors",
                confidence=1.0
            ))
        except Exception as e:
            self.logger.warning(f"Error analyzing JSON: {str(e)}")
        
        return threats
    
    def _get_json_depth(self, obj, current_depth=0):
        """Calculate maximum depth of JSON object"""
        if isinstance(obj, dict):
            return max([self._get_json_depth(v, current_depth + 1) for v in obj.values()], default=current_depth)
        elif isinstance(obj, list):
            return max([self._get_json_depth(item, current_depth + 1) for item in obj], default=current_depth)
        else:
            return current_depth
    
    def _find_large_arrays(self, obj, path="root"):
        """Find large arrays in JSON structure"""
        large_arrays = []
        
        if isinstance(obj, list):
            if len(obj) > 10000:
                large_arrays.append((path, len(obj)))
            for i, item in enumerate(obj[:10]):  # Check first 10 items
                large_arrays.extend(self._find_large_arrays(item, f"{path}[{i}]"))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                large_arrays.extend(self._find_large_arrays(value, f"{path}.{key}"))
        
        return large_arrays
    
    async def _llm_security_analysis(self, file_content: bytes, file_info: Dict[str, Any], metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze file content for security issues"""
        if not self.llm_client:
            return None
        
        try:
            # Prepare content sample for LLM analysis
            text_content = file_content.decode('utf-8', errors='ignore')
            
            # Limit content size for LLM analysis
            max_chars = 5000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars] + "\n... [truncated]"
            
            analysis_prompt = self._create_security_analysis_prompt(text_content, file_info, metadata)
            
            llm_response = await self.llm_client.generate_response(
                prompt=analysis_prompt,
                max_tokens=800,
                temperature=0.1
            )
            
            return self._parse_llm_security_response(llm_response)
            
        except Exception as e:
            self.logger.error(f"LLM security analysis failed: {str(e)}")
            return None
    
    def _create_security_analysis_prompt(self, content: str, file_info: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Create prompt for LLM security analysis"""
        return f"""
Analyze this file content for security threats and privacy concerns:

File Information:
- Name: {file_info.get('name', 'unknown')}
- Type: {file_info.get('extension', 'unknown')}
- Size: {file_info.get('size_mb', 0):.2f}MB

Content Sample:
{content}

Please analyze for:
1. Malicious code patterns (scripts, injections, system calls)
2. Sensitive data (PII, credentials, financial info)
3. Suspicious data structures or patterns
4. Privacy violations
5. Data integrity issues

Respond in JSON format:
{{
    "overall_risk": "low/medium/high/critical",
    "threats": [
        {{
            "type": "threat_type",
            "severity": "low/medium/high/critical",
            "description": "detailed description",
            "location": "where found",
            "recommendation": "how to fix",
            "confidence": 0.8
        }}
    ],
    "privacy_concerns": ["list of privacy issues"],
    "recommendations": ["security recommendations"]
}}
"""
    
    def _parse_llm_security_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM security analysis response"""
        try:
            # Try to parse as JSON
            response_data = json.loads(llm_response.strip())
            return response_data
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse as text
            return {
                "overall_risk": "medium",
                "threats": [],
                "privacy_concerns": [],
                "recommendations": ["Manual review recommended"],
                "raw_response": llm_response
            }
    
    def _parse_llm_threats(self, llm_threats: List[Dict[str, Any]]) -> List[SecurityThreat]:
        """Parse LLM threats into SecurityThreat objects"""
        threats = []
        
        for threat_data in llm_threats:
            try:
                # Map threat type
                threat_type_map = {
                    'malicious_code': ThreatType.MALICIOUS_CODE,
                    'data_injection': ThreatType.DATA_INJECTION,
                    'privacy_violation': ThreatType.PRIVACY_VIOLATION,
                    'suspicious_pattern': ThreatType.SUSPICIOUS_PATTERN,
                    'invalid_format': ThreatType.INVALID_FORMAT
                }
                
                threat_type = threat_type_map.get(
                    threat_data.get('type', '').lower(),
                    ThreatType.SUSPICIOUS_PATTERN
                )
                
                # Map severity
                severity_map = {
                    'low': SecurityLevel.LOW,
                    'medium': SecurityLevel.MEDIUM,
                    'high': SecurityLevel.HIGH,
                    'critical': SecurityLevel.CRITICAL
                }
                
                severity = severity_map.get(
                    threat_data.get('severity', 'medium').lower(),
                    SecurityLevel.MEDIUM
                )
                
                threat = SecurityThreat(
                    threat_type=threat_type,
                    severity=severity,
                    description=threat_data.get('description', 'LLM detected threat'),
                    location=threat_data.get('location', 'Unknown'),
                    recommendation=threat_data.get('recommendation', 'Review manually'),
                    confidence=float(threat_data.get('confidence', 0.5)),
                    metadata={'source': 'llm_analysis'}
                )
                
                threats.append(threat)
                
            except Exception as e:
                self.logger.warning(f"Error parsing LLM threat: {str(e)}")
        
        return threats

class AccessControlManager:
    """Manage access control and session isolation"""
    
    def __init__(self):
        self.session_permissions: Dict[str, Dict[str, Any]] = {}
        self.access_logs: List[Dict[str, Any]] = []
        self.blocked_sessions: set = set()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, session_id: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create new session with permissions"""
        permissions = {
            'file_upload': True,
            'data_analysis': True,
            'agent_access': True,
            'max_file_size_mb': 100,
            'max_concurrent_tasks': 10,
            'allowed_file_types': ['.csv', '.json', '.xlsx', '.txt'],
            'created_at': datetime.now(),
            'user_context': user_context or {}
        }
        
        self.session_permissions[session_id] = permissions
        
        self._log_access(session_id, 'session_created', {'permissions': permissions})
        
        return permissions
    
    def check_permission(self, session_id: str, action: str, resource: str = None) -> bool:
        """Check if session has permission for action"""
        if session_id in self.blocked_sessions:
            return False
        
        if session_id not in self.session_permissions:
            return False
        
        permissions = self.session_permissions[session_id]
        
        # Check specific permissions
        if action == 'file_upload':
            return permissions.get('file_upload', False)
        elif action == 'data_analysis':
            return permissions.get('data_analysis', False)
        elif action == 'agent_access':
            return permissions.get('agent_access', False)
        
        return True
    
    def check_rate_limit(self, session_id: str, max_requests: int = 60, window_minutes: int = 1) -> bool:
        """Check rate limiting for session"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)
        
        if session_id not in self.rate_limits:
            self.rate_limits[session_id] = []
        
        # Remove old requests
        self.rate_limits[session_id] = [
            timestamp for timestamp in self.rate_limits[session_id]
            if timestamp > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[session_id]) >= max_requests:
            self._log_access(session_id, 'rate_limit_exceeded', {
                'requests_in_window': len(self.rate_limits[session_id]),
                'max_requests': max_requests
            })
            return False
        
        # Add current request
        self.rate_limits[session_id].append(now)
        return True
    
    def validate_file_access(self, session_id: str, file_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate file access permissions"""
        if not self.check_permission(session_id, 'file_upload'):
            return False, "File upload not permitted"
        
        permissions = self.session_permissions.get(session_id, {})
        
        # Check file size
        file_size_mb = file_info.get('size_mb', 0)
        max_size = permissions.get('max_file_size_mb', 50)
        if file_size_mb > max_size:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {max_size}MB"
        
        # Check file type
        file_extension = file_info.get('extension', '').lower()
        allowed_types = permissions.get('allowed_file_types', [])
        if file_extension not in allowed_types:
            return False, f"File type '{file_extension}' not allowed"
        
        return True, "Access granted"
    
    def block_session(self, session_id: str, reason: str):
        """Block session access"""
        self.blocked_sessions.add(session_id)
        self._log_access(session_id, 'session_blocked', {'reason': reason})
        self.logger.warning(f"Session {session_id} blocked: {reason}")
    
    def unblock_session(self, session_id: str):
        """Unblock session access"""
        self.blocked_sessions.discard(session_id)
        self._log_access(session_id, 'session_unblocked', {})
        self.logger.info(f"Session {session_id} unblocked")
    
    def _log_access(self, session_id: str, action: str, metadata: Dict[str, Any]):
        """Log access event"""
        log_entry = {
            'timestamp': datetime.now(),
            'session_id': session_id,
            'action': action,
            'metadata': metadata
        }
        
        self.access_logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]
    
    def get_access_logs(self, session_id: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get access logs"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        logs = [
            log for log in self.access_logs
            if log['timestamp'] > cutoff
        ]
        
        if session_id:
            logs = [log for log in logs if log['session_id'] == session_id]
        
        return logs
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        return {
            'active_sessions': len(self.session_permissions),
            'blocked_sessions': len(self.blocked_sessions),
            'total_access_logs': len(self.access_logs),
            'recent_blocks': len([
                log for log in self.access_logs[-100:]
                if log['action'] == 'session_blocked'
            ]),
            'rate_limit_violations': len([
                log for log in self.access_logs[-100:]
                if log['action'] == 'rate_limit_exceeded'
            ])
        }

class LLMSecurityValidator:
    """Main LLM-enhanced security validation system"""
    
    def __init__(self, llm_client=None):
        self.file_analyzer = FileSecurityAnalyzer(llm_client)
        self.access_manager = AccessControlManager()
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.logger = logging.getLogger(__name__)
    
    async def validate_file_upload(self, 
                                 session_id: str,
                                 file_path: str,
                                 file_content: bytes = None,
                                 metadata: Dict[str, Any] = None) -> ValidationResult:
        """Comprehensive file upload validation"""
        
        try:
            # Check session permissions
            file_info = self.file_analyzer._get_file_info(file_path, file_content)
            access_granted, access_message = self.access_manager.validate_file_access(session_id, file_info)
            
            if not access_granted:
                return ValidationResult(
                    is_safe=False,
                    threats=[SecurityThreat(
                        threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                        severity=SecurityLevel.HIGH,
                        description=access_message,
                        location=session_id,
                        recommendation="Check session permissions",
                        confidence=1.0
                    )],
                    file_info=file_info,
                    processing_time=0.0
                )
            
            # Check rate limiting
            if not self.access_manager.check_rate_limit(session_id):
                return ValidationResult(
                    is_safe=False,
                    threats=[SecurityThreat(
                        threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                        severity=SecurityLevel.MEDIUM,
                        description="Rate limit exceeded",
                        location=session_id,
                        recommendation="Wait before uploading more files",
                        confidence=1.0
                    )],
                    file_info=file_info,
                    processing_time=0.0
                )
            
            # Check cache for previous analysis
            file_hash = file_info.get('hash')
            if file_hash and file_hash in self.validation_cache:
                cached_result = self.validation_cache[file_hash]
                self.logger.info(f"Using cached validation result for {file_path}")
                return cached_result
            
            # Perform security analysis
            result = await self.file_analyzer.analyze_file(file_path, file_content, metadata)
            
            # Cache result
            if file_hash:
                self.validation_cache[file_hash] = result
            
            # Log validation
            self.access_manager._log_access(session_id, 'file_validated', {
                'file_path': file_path,
                'is_safe': result.is_safe,
                'threat_count': len(result.threats),
                'processing_time': result.processing_time
            })
            
            # Block session if critical threats found
            critical_threats = [t for t in result.threats if t.severity == SecurityLevel.CRITICAL]
            if critical_threats:
                self.access_manager.block_session(
                    session_id,
                    f"Critical security threats detected in uploaded file"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            
            return ValidationResult(
                is_safe=False,
                threats=[SecurityThreat(
                    threat_type=ThreatType.INVALID_FORMAT,
                    severity=SecurityLevel.HIGH,
                    description=f"Validation error: {str(e)}",
                    location=file_path,
                    recommendation="Contact support",
                    confidence=1.0
                )],
                file_info={'error': str(e)},
                processing_time=0.0
            )
    
    def create_session(self, session_id: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create secure session"""
        return self.access_manager.create_session(session_id, user_context)
    
    def get_security_recommendations(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Get security recommendations"""
        recommendations = []
        
        # Check recent validation results
        recent_threats = []
        for result in self.validation_cache.values():
            recent_threats.extend(result.threats)
        
        # Analyze threat patterns
        threat_types = {}
        for threat in recent_threats:
            threat_types[threat.threat_type] = threat_types.get(threat.threat_type, 0) + 1
        
        # Generate recommendations based on patterns
        if ThreatType.MALICIOUS_CODE in threat_types:
            recommendations.append({
                'type': 'malicious_code_detection',
                'priority': 'high',
                'title': 'Malicious Code Detected',
                'description': f'{threat_types[ThreatType.MALICIOUS_CODE]} files contained malicious patterns',
                'actions': [
                    'Implement stricter file filtering',
                    'Enable real-time scanning',
                    'Review file sources'
                ]
            })
        
        if ThreatType.PRIVACY_VIOLATION in threat_types:
            recommendations.append({
                'type': 'privacy_concerns',
                'priority': 'medium',
                'title': 'Privacy Violations Detected',
                'description': f'{threat_types[ThreatType.PRIVACY_VIOLATION]} files contained sensitive data',
                'actions': [
                    'Implement data anonymization',
                    'Add privacy warnings',
                    'Review data handling procedures'
                ]
            })
        
        # Check access patterns
        access_logs = self.access_manager.get_access_logs(session_id, hours=24)
        blocked_count = len([log for log in access_logs if log['action'] == 'session_blocked'])
        
        if blocked_count > 0:
            recommendations.append({
                'type': 'access_violations',
                'priority': 'high',
                'title': 'Access Violations',
                'description': f'{blocked_count} sessions were blocked in the last 24 hours',
                'actions': [
                    'Review access control policies',
                    'Investigate blocked sessions',
                    'Enhance monitoring'
                ]
            })
        
        return recommendations
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard"""
        
        # Validation statistics
        total_validations = len(self.validation_cache)
        safe_files = sum(1 for result in self.validation_cache.values() if result.is_safe)
        
        # Threat statistics
        all_threats = []
        for result in self.validation_cache.values():
            all_threats.extend(result.threats)
        
        threat_by_type = {}
        threat_by_severity = {}
        
        for threat in all_threats:
            threat_by_type[threat.threat_type.value] = threat_by_type.get(threat.threat_type.value, 0) + 1
            threat_by_severity[threat.severity.value] = threat_by_severity.get(threat.severity.value, 0) + 1
        
        return {
            'validation_stats': {
                'total_files_validated': total_validations,
                'safe_files': safe_files,
                'unsafe_files': total_validations - safe_files,
                'safety_rate': safe_files / total_validations if total_validations > 0 else 0
            },
            'threat_analysis': {
                'total_threats': len(all_threats),
                'threats_by_type': threat_by_type,
                'threats_by_severity': threat_by_severity
            },
            'access_control': self.access_manager.get_security_summary(),
            'recommendations': self.get_security_recommendations()
        }

# Global security validator instance
security_validator = LLMSecurityValidator()