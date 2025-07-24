#!/usr/bin/env python3
"""
Report Generator Debug Test
"""

import requests
import json
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_report_debug():
    # Report Generator 서버 시작
    logger.info('🚀 Starting Report Generator server...')
    process = subprocess.Popen([sys.executable, 'a2a_ds_servers/report_generator_server_new.py'], 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(8)  # 더 긴 시간 대기

    try:
        # 테스트 데이터
        test_data = {
            'jsonrpc': '2.0',
            'method': 'message/send',
            'params': {
                'message': {
                    'messageId': 'debug_test',
                    'role': 'user',
                    'parts': [
                        {
                            'kind': 'text',
                            'text': 'quarter,revenue,profit\nQ1,100000,15000\nQ2,120000,18000\n\nGenerate a business report'
                        }
                    ]
                }
            },
            'id': 'debug_test'
        }
        
        # 테스트 요청
        logger.info('📤 Sending test request...')
        response = requests.post('http://localhost:8316/', json=test_data, timeout=30)
        logger.info(f'Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            logger.info('✅ Request successful!')
            logger.info(f'Response length: {len(str(result))} chars')
        else:
            logger.error(f'❌ Request failed: {response.text}')
            
    except Exception as e:
        logger.error(f'Error during request: {e}')
        
    finally:
        # 서버 로그 출력
        try:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                logger.info('📋 Server logs:')
                # 마지막 부분만 출력
                lines = stdout.split('\n')
                for line in lines[-50:]:  # 마지막 50줄
                    if line.strip():
                        print(line)
        except:
            logger.warning('Could not get server logs')
            
        logger.info('🛑 Server stopped')

if __name__ == "__main__":
    test_report_debug()