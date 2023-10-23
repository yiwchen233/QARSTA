__all__ = ['ParameterList']


class ParameterList(object):
    def __init__(self, n):
        self.n = n
        self.params = {}
        # Criticality constants
        self.params["general.criticality_step_eps"] = 1e-12
        self.params["general.criticality_step_mu"] = 1e12
        # Geometry of sample set
        self.params["geometry.sample_set_radius_tol"] = 10.0
        self.params["geometry.tol"] = 1e-6
        # Trust Region Radius management
        self.params["tr_radius.eta1"] = 0.1
        self.params["tr_radius.eta2"] = 0.7
        self.params["tr_radius.gamma_dec"] = 0.5
        self.params["tr_radius.gamma_inc"] = 2.0
        self.params["tr_radius.gamma_inc_overline"] = 4.0
        self.params["tr_radius.delta_max"] = 1.0e10
        self.params["tr_radius.alpha1"] = 0.1
        self.params["tr_radius.alpha2"] = 0.5
        # Accuracy level
        self.params["model.rel_tol"] = 1e-20

        self.params_changed = {}
        for p in self.params:
            self.params_changed[p] = False

    def __call__(self, key, new_value=None):  # self(key) or self(key, new_value)
        if key in self.params:
            if new_value is None:
                return self.params[key]
            else:
                if self.params_changed[key]:
                    raise ValueError("Trying to update parameter '%s' for a second time" % key)
                self.params[key] = new_value
                self.params_changed[key] = True
                return self.params[key]
        else:
            raise ValueError("Unknown parameter '%s'" % key)

    def param_type(self, key):
        # Use the check_* methods below, but switch based on key
        if key == "general.criticality_step_eps":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "general.criticality_step_mu":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "geometry.sample_set_radius_tol":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None  
        elif key == "geometry.tol":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "tr_radius.eta1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.eta2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_dec":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.gamma_inc":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.gamma_inc_overline":
            type_str, nonetype_ok, lower, upper = 'float', False, 1.0, None
        elif key == "tr_radius.delta_max":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, None
        elif key == "tr_radius.alpha1":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "tr_radius.alpha2":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        elif key == "model.rel_tol":
            type_str, nonetype_ok, lower, upper = 'float', False, 0.0, 1.0
        else:
            assert False, "ParameterList.param_type() has unknown key: %s" % key
        return type_str, nonetype_ok, lower, upper

    def check_param(self, key, value):
        type_str, nonetype_ok, lower, upper = self.param_type(key)
        if type_str == 'float':
            return check_float(value, lower=lower, upper=upper, allow_nonetype=nonetype_ok)
        else:
            assert False, "Unknown type_str '%s' for parameter '%s'" % (type_str, key)

    def check_all_params(self):
        bad_keys = []
        for key in self.params:
            if not self.check_param(key, self.params[key]):
                bad_keys.append(key)
        return len(bad_keys) == 0, bad_keys


def check_float(val, lower=None, upper=None, allow_nonetype=False):
    # Check that val is a float and (optionally) that lower <= val <= upper
    if val is None:
        return allow_nonetype
    elif not isinstance(val, float):
        return False
    else:  # is integer
        return (lower is None or val >= lower) and (upper is None or val <= upper)

