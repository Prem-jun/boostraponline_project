def add_exclamination(func): # decorator function
    def wrapper(name):
        return func(name)+'!'
    return wrapper

def add_symbol(symbol):
    def decorator(func):
        def wrapper(name):
            return func(name)+symbol
        return wrapper
    return decorator

@add_exclamination    
def greet(name): # function is decorated
    return 'hello '+ name

print(greet('prem'))

@add_symbol("###")
def greet1(name):
    return 'hell0 '+name

print(greet1('prem'))