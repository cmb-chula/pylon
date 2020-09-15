class Stateful:
    """implement resumable classes,
    resumables are kept in _state dictionary,
    it is allowed to be nested (automatically detected), tracable through _children
    """
    def __init__(self):
        super(Stateful, self).__setattr__('_state', {})
        super(Stateful, self).__setattr__('_children', {})

    def get_state(self):
        state = {'self': self._state}
        # get states from the chlidren
        state.update({k: v.get_state() for k, v in self._children.items()})
        return state

    def load_state(self, state):
        self._state = state['self']
        # load states for the children
        for k, v in self._children.items():
            v.load_state(state[k])

    def __getattribute__(self, name):
        """allows accessing keys in _state via class attributes
        i.e. self.avg might lead to self._state['avg'] if it exists
        however, it will try to find the true self.avg first
        """
        try:
            return super(Stateful, self).__getattribute__(name)
        except AttributeError:
            try:
                # try state keys
                state = super(Stateful, self).__getattribute__('_state')
                return state[name]
            except KeyError:
                raise AttributeError(f'attribute: {name}')

    def __setattr__(self, name, value):
        # try to warn the user to call super()
        assert hasattr(self, '_state'), "you must call super()"
        if isinstance(value, Stateful):
            # register the children
            self._children[name] = value
            return super(Stateful, self).__setattr__(name, value)
        elif name in self.__dict__:
            # if changing existing attribute
            return super(Stateful, self).__setattr__(name, value)
        elif name in self._state:
            # if it is in the state
            self._state[name] = value
        else:
            return super(Stateful, self).__setattr__(name, value)

    def is_state_empty(self):
        """check if the state is empty, 
        if the state is not altered, it is a sign that this class doesn't need a state"""
        state = self.get_state()
        # if it only has "self" and nothing in it
        return len(state) == 1 and len(state['self']) == 0
