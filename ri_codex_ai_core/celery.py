import os
from celery import Celery


# Set the default Django settings module for Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ri_codex_ai_core.settings')

# Create a Celery instance
app = Celery('ri_codex_ai_core')

# Load configuration from Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')

# Discover and register task modules in Django apps
app.autodiscover_tasks()

app.conf.update(
    CELERY_WORKER_HIJACK_ROOT_LOGGER=False,
    task_track_started=True,
    task_track_tracebacks=True
)

