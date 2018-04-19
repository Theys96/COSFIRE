import cosfire

class ImageStack():

    def __init__(self, filter, filterArgs):
        for arg in filterArgs:
            print(arg)

stack = ImageStack(cosfire.DoGFilter, (2.6, 1))
