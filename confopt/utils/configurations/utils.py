def create_config_hash(config: dict) -> str:
    """Create a fast hashable representation of a configuration"""
    items = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, (int, float, bool)):
            items.append(f"{k}:{v}")
        else:
            items.append(f"{k}:{str(v)}")
    return "|".join(items)
