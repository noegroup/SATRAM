class EmptyProgress():
    def __call__(self, iterable):
        return iter(iterable)

def handle_progress(progress):
    if progress is None:
        return EmptyProgress()
    else:
        return progress