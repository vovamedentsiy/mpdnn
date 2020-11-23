def piece_wise_linear_fn(x, y_0,y_1,x_0,x_1):
    '''
    This fn computes y_0 + (y_1 - y_0)/(x_1-x_0)*(x - x_0),
    i.e. a fn with offset y_0 and slope (y_1 - y_0)/(x_1-x_0)
    before x_0 the value is returned as y_0
    after x_1 the value is returned as y_1
    :param y_0:
    :param y_1:
    :param x_0:
    :param x__1:
    :return:
    '''
    x = max(x,x_0)
    x = min(x,x_1)
    if x_1 == x_0:
        return y_1
    return y_0 + ((y_1-y_0)/(x_1-x_0))*(x-x_0)