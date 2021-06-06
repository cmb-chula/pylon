def rename(newname):
    """define the name of the function"""
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator