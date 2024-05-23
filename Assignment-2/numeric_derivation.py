
def derive(f, x, h=0.0001):
    # return None  # TODO: implement this function 
    return (f(x + h) - f(x - h)) / (2 * h)
