def create_config_hash(config: dict) -> int:
    """Create a fast hashable representation of a configuration using tuples."""
    items = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, (int, float, bool)):
            items.append((k, v))
        else:
            items.append((k, str(v)))
    return hash(tuple(items))
