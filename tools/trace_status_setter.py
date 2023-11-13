from opentelemetry.trace.status import StatusCode
def status_setter (span, value):
    if value:
        status_code=StatusCode(1)
    else:
        status_code=StatusCode(2)
    span.set_status(status_code)