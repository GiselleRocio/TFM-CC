import uuid


def new_run_uuid() -> str:
    """Genera un UUID4 único para identificar una corrida completa."""
    return str(uuid.uuid4())
