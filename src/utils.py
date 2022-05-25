def get_argparse_defaults(parser):
    """ Get hold of the default arguments without having to parse first

    Args:
        parser (argparse.ArgumentParser): Object for parsing command line strings into Python objects.

    Returns:
        [dict]: Dictionary of default arguments
    """    
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults

class Map(dict):
    """dot.notation access to dictionary attributes
        Example:
        m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

    Args:
        dict (dict): Dictionary to map
    """   
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]