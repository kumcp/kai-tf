
class DataExtraction():

    @staticmethod
    def dict_2_obj(dict_var={}):
        return Struct(dict_var)


class Struct():
    def __init__(self, dict_var={}):
        self.dict_var = dict_var

    def __getattr__(self, var_name):
        if var_name in self.dict_var:
            return self.dict_var[var_name]
        else:
            return 0

    def __repr__(self):
        return self.dict_var

    def __str__(self):
        return str(self.dict_var)
