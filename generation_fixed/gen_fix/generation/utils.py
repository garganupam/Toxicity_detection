from contextlib import contextmanager

@contextmanager
def temporary_eval(model):
    """
    Temporarily set a PyTorch model to eval mode, then restore its original state.
    Usage:
        with temporary_eval(model):
            ... # model is in eval()
        # model is restored to its original training/eval state
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()
