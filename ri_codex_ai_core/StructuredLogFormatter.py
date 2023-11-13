from datetime import datetime
import logging
import json
class StructuredLogFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'user_id': record.__dict__.get('user_id', ''),
            'trace_id': record.__dict__.get('trace_id', ''),
            'span_id': record.__dict__.get('span_id', ''),
            'module': record.module,
            'message': record.getMessage()
        }
        return json.dumps(log_data)
