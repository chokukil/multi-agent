# A2A Data Science Servers - Logging Utilities
# Enhanced logging for A2A protocol with structured output

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import traceback

def setup_a2a_logger(
    name: str = "a2a_ds_server",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True
) -> logging.Logger:
    """
    Setup a structured logger for A2A Data Science servers.
    
    Parameters:
    ----------
    name : str, optional
        Logger name, by default "a2a_ds_server".
    log_level : str, optional
        Logging level, by default "INFO".
    log_file : str, optional
        Log file path, by default None (stdout only).
    structured : bool, optional
        Whether to use structured JSON logging, by default True.
        
    Returns:
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    """
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'agent_name'):
            log_data['agent_name'] = record.agent_name
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'context_id'):
            log_data['context_id'] = record.context_id
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str)

async def log_agent_execution_dynamic(
    logger: logging.Logger,
    agent_name: str,
    operation: str,
    status: str,
    execution_time: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    user_query: Optional[str] = None
) -> None:
    """
    LLM First 동적 에이전트 실행 로깅
    LLM이 로그 형식, 레벨, 메시지를 동적으로 결정
    """
    
    # LLM 사용 가능 여부 확인
    try:
        from openai import AsyncOpenAI
        import os
        
        llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        LLM_AVAILABLE = True
    except:
        LLM_AVAILABLE = False
    
    if LLM_AVAILABLE:
        try:
            # LLM이 로깅 컨텍스트를 동적으로 분석
            context_prompt = f"""
당신은 시스템 로깅 전문가입니다. 다음 에이전트 실행 정보를 분석하여 최적의 로깅 방식을 결정해주세요.

에이전트 정보:
- 에이전트명: {agent_name}
- 작업: {operation}
- 상태: {status}
- 실행 시간: {execution_time}초 (있는 경우)
- 사용자 질문: {user_query}
- 상세 정보: {details}

이 상황에 맞는 로그 레벨(INFO/WARNING/ERROR), 메시지 형식, 중요도를 결정하고
다음 JSON 형식으로 응답해주세요:
{{
    "log_level": "INFO/WARNING/ERROR",
    "message": "상황에 맞는 로그 메시지",
    "importance": "high/medium/low",
    "category": "로그 카테고리",
    "tags": ["관련 태그들"]
}}

템플릿이나 고정된 형식을 사용하지 말고, 이 특정 상황에 맞는 최적의 로깅을 제안해주세요.
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 시스템 로깅 전문가입니다."},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.3
            )
            
            # LLM 응답 파싱
            import json
            import re
            
            response_content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            
            if json_match:
                llm_decision = json.loads(json_match.group())
                
                # LLM 결정에 따른 동적 로깅
                log_level = llm_decision.get("log_level", "INFO")
                message = llm_decision.get("message", f"Agent {agent_name} {operation} - {status}")
                importance = llm_decision.get("importance", "medium")
                category = llm_decision.get("category", "agent_execution")
                tags = llm_decision.get("tags", [])
                
                # 동적 extra 필드 생성
                extra = {
                    'agent_name': agent_name,
                    'operation': operation,
                    'status': status,
                    'llm_determined_importance': importance,
                    'llm_category': category,
                    'llm_tags': tags,
                    'dynamic_logging': True
                }
                
                if execution_time is not None:
                    extra['execution_time'] = execution_time
                
                if task_id:
                    extra['task_id'] = task_id
                
                if context_id:
                    extra['context_id'] = context_id
                
                if details:
                    extra.update(details)
                
                # LLM이 결정한 로그 레벨로 기록
                if log_level == "ERROR":
                    logger.error(message, extra=extra)
                elif log_level == "WARNING":
                    logger.warning(message, extra=extra)
                else:
                    logger.info(message, extra=extra)
                
                return
                
        except Exception as e:
            # LLM 처리 실패 시 기본 로깅으로 폴백
            logger.warning(f"LLM 동적 로깅 실패, 기본 로깅으로 폴백: {e}")
    
    # 기본 로깅 (LLM 없거나 실패 시)
    message = f"Agent {agent_name} {operation} - {status}"
    
    extra = {
        'agent_name': agent_name,
        'operation': operation,
        'status': status,
        'dynamic_logging': False
    }
    
    if execution_time is not None:
        extra['execution_time'] = execution_time
        message += f" ({execution_time:.2f}s)"
    
    if task_id:
        extra['task_id'] = task_id
    
    if context_id:
        extra['context_id'] = context_id
    
    if details:
        extra.update(details)
    
    # 기본 로그 레벨 결정
    if status == "failed":
        logger.error(message, extra=extra)
    elif status == "started":
        logger.info(message, extra=extra)
    else:
        logger.info(message, extra=extra)

def log_agent_execution(
    logger: logging.Logger,
    agent_name: str,
    operation: str,
    status: str,
    execution_time: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None
) -> None:
    """
    레거시 호환성을 위한 래퍼 함수
    내부적으로 동적 로깅을 사용하도록 변경
    """
    import asyncio
    
    # 비동기 동적 로깅 호출
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 태스크로 실행
            loop.create_task(log_agent_execution_dynamic(
                logger, agent_name, operation, status, 
                execution_time, details, task_id, context_id
            ))
        else:
            # 새로운 이벤트 루프 실행
            asyncio.run(log_agent_execution_dynamic(
                logger, agent_name, operation, status, 
                execution_time, details, task_id, context_id
            ))
    except Exception as e:
        # 비동기 처리 실패 시 기본 로깅으로 폴백
        logger.warning(f"동적 로깅 실행 실패, 기본 로깅 사용: {e}")
        
        message = f"Agent {agent_name} {operation} - {status}"
        
        extra = {
            'agent_name': agent_name,
            'operation': operation,
            'status': status,
        }
        
        if execution_time is not None:
            extra['execution_time'] = execution_time
            message += f" ({execution_time:.2f}s)"
        
        if task_id:
            extra['task_id'] = task_id
        
        if context_id:
            extra['context_id'] = context_id
        
        if details:
            extra.update(details)
        
        # 기본 로그 레벨
        if status == "failed":
            logger.error(message, extra=extra)
        elif status == "started":
            logger.info(message, extra=extra)
        else:
            logger.info(message, extra=extra)

def log_ai_function(
    response: str,
    file_name: str,
    log: bool = True,
    log_path: str = './logs/',
    overwrite: bool = True,
    agent_name: str = "unknown_agent",
    metadata: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Enhanced version of the original log_ai_function with A2A protocol support.
    
    Parameters
    ----------
    response : str
        The response of the AI function.
    file_name : str
        The name of the file to save the response to.
    log : bool, optional
        Whether to log the response or not. The default is True.
    log_path : str, optional
        The path to save the log file. The default is './logs/'.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. The default is True.
    agent_name : str, optional
        Name of the agent generating the response.
    metadata : Dict[str, Any], optional
        Additional metadata to include.
    
    Returns
    -------
    tuple
        The path and name of the log file.    
    """
    
    if not log:
        return (None, None)
    
    # Ensure the directory exists
    os.makedirs(log_path, exist_ok=True)
    
    file_path = os.path.join(log_path, file_name)
    
    if not overwrite and os.path.exists(file_path):
        # Generate unique filename
        base_name, ext = os.path.splitext(file_name)
        i = 1
        while True:
            new_file_name = f"{base_name}_{i}{ext}"
            new_file_path = os.path.join(log_path, new_file_name)
            if not os.path.exists(new_file_path):
                file_path = new_file_path
                file_name = new_file_name
                break
            i += 1
    
    # Prepare content with metadata
    content_with_metadata = prepare_content_with_metadata(
        response, agent_name, metadata
    )
    
    # Write the file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content_with_metadata)
        
        print(f"📁 File saved to: {file_path}")
        return (file_path, file_name)
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return (None, None)

def prepare_content_with_metadata(
    content: str,
    agent_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare content with A2A protocol metadata header.
    
    Parameters:
    ----------
    content : str
        Original content.
    agent_name : str
        Agent name.
    metadata : Dict[str, Any], optional
        Additional metadata.
        
    Returns:
    -------
    str
        Content with metadata header.
    """
    timestamp = datetime.now().isoformat()
    
    header_lines = [
        "# A2A Data Science Agent Generated Content",
        f"# Agent: {agent_name}",
        f"# Timestamp: {timestamp}",
        "# Protocol: A2A v0.2.9",
        "# ======================================",
        "",
    ]
    
    if metadata:
        header_lines.insert(-2, f"# Metadata: {json.dumps(metadata, default=str)}")
    
    header = "\n".join(header_lines)
    return header + content

def create_execution_log(
    agent_name: str,
    operation: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    execution_time: float,
    status: str = "completed",
    errors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive execution log entry.
    
    Parameters:
    ----------
    agent_name : str
        Name of the executing agent.
    operation : str
        Operation performed.
    input_data : Dict[str, Any]
        Input data (sanitized).
    output_data : Dict[str, Any]
        Output data (sanitized).
    execution_time : float
        Execution time in seconds.
    status : str, optional
        Execution status.
    errors : List[str], optional
        List of error messages.
        
    Returns:
    -------
    Dict[str, Any]
        Comprehensive execution log.
    """
    log_entry = {
        "agent_name": agent_name,
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "status": status,
        "input_summary": {
            "keys": list(input_data.keys()) if isinstance(input_data, dict) else [],
            "size": len(str(input_data)),
        },
        "output_summary": {
            "keys": list(output_data.keys()) if isinstance(output_data, dict) else [],
            "size": len(str(output_data)),
        },
    }
    
    if errors:
        log_entry["errors"] = errors
    
    return log_entry

def log_artifact_creation(
    logger: logging.Logger,
    artifact_type: str,
    filename: str,
    size: int,
    agent_name: str,
    success: bool = True
) -> None:
    """
    Log artifact creation events.
    
    Parameters:
    ----------
    logger : logging.Logger
        Logger instance.
    artifact_type : str
        Type of artifact (plot, data, code, etc.).
    filename : str
        Generated filename.
    size : int
        Artifact size in bytes.
    agent_name : str
        Agent that created the artifact.
    success : bool, optional
        Whether creation was successful.
    """
    message = f"Artifact created: {filename} ({artifact_type}, {size} bytes)"
    
    extra = {
        'agent_name': agent_name,
        'artifact_type': artifact_type,
        'filename': filename,
        'size': size,
        'success': success,
    }
    
    if success:
        logger.info(message, extra=extra)
    else:
        logger.error(f"Failed to create artifact: {filename}", extra=extra)

def save_execution_summary(
    log_path: str,
    execution_logs: List[Dict[str, Any]],
    summary_filename: str = "execution_summary.json"
) -> str:
    """
    Save execution summary to file.
    
    Parameters:
    ----------
    log_path : str
        Path to save the summary.
    execution_logs : List[Dict[str, Any]]
        List of execution log entries.
    summary_filename : str, optional
        Summary filename.
        
    Returns:
    -------
    str
        Path to saved summary file.
    """
    os.makedirs(log_path, exist_ok=True)
    
    summary = {
        "summary_timestamp": datetime.now().isoformat(),
        "total_executions": len(execution_logs),
        "successful_executions": len([log for log in execution_logs if log.get("status") == "completed"]),
        "failed_executions": len([log for log in execution_logs if log.get("status") == "failed"]),
        "total_execution_time": sum(log.get("execution_time", 0) for log in execution_logs),
        "agents_used": list(set(log.get("agent_name") for log in execution_logs)),
        "executions": execution_logs,
    }
    
    summary_path = os.path.join(log_path, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary_path 