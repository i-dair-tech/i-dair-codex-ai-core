from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, TraceFlags

def get_context(request):
    if 'Trace-Id' in request.headers and 'Span-Id' in request.headers:
        trace_id=request.headers["Trace-Id"]
        span_id=request.headers["Span-Id"]
        
        span_context = trace.SpanContext(
            trace_id=int(trace_id,16),
            span_id=int(span_id,16),
            is_remote=True, 
            trace_flags=TraceFlags(0x01)
            )
    else:
        span_context = trace.SpanContext(
            trace_id=0,
            span_id=0,
            is_remote=True, 
            trace_flags=TraceFlags(0x01)
            )
    context = trace.set_span_in_context(NonRecordingSpan(span_context))
    return context