def get_parent_class(x, converter):
    """
    Helper function to get.
    - x: 
    - converter: dictionary of values to pairs
    """
    if "," in x:
        genres = x.split(",")
        new_genres = set(str(converter[int(c)]) for c in genres)
        return int(new_genres.pop())
    else:
        if((len(x)==0) | (x == " ")):
            # There are some missing values
            return(1000)
        else:
            return converter[int(x)]