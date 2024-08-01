from math import sqrt

def Distance_Between(A, B):
    # Unpack the coordinates
    x1 = A.x
    y1 = A.y
    
    x2 = B.x
    y2 = B.y
    
    
    # Calculate the squared differences
    squared_diff_x = (x2 - x1) ** 2
    squared_diff_y = (y2 - y1) ** 2

    # Sum the squared differences and take the square root
    distance = sqrt(squared_diff_x + squared_diff_y)
    
    return distance