__all__ = ['EXIT_MAXFUN_WARNING', 'EXIT_SUCCESS', 'EXIT_INPUT_ERROR', 'EXIT_TR_INCREASE_ERROR', 'EXIT_LINALG_ERROR', 'ExitInformation', 'OptimResults']

EXIT_MAXFUN_WARNING = 1  # warning, reached max function evals
EXIT_SUCCESS = 0  # successful finish (delta=deltaend, sufficient objective reduction, or everything in noise level)
EXIT_INPUT_ERROR = -1  # error, bad inputs
EXIT_TR_INCREASE_ERROR = -2  # error, trust region step increased model value
EXIT_LINALG_ERROR = -3  # error, linalg error (singular matrix encountered)


class ExitInformation:
    def __init__(self, flag, msg_details):
        self.flag = flag
        self.msg = msg_details

    def message(self, with_stem=True):
        if not with_stem:
            return self.msg
        elif self.flag == EXIT_SUCCESS:
            return "Success: " + self.msg
        elif self.flag == EXIT_MAXFUN_WARNING:
            return "Warning (max evals): " + self.msg
        elif self.flag == EXIT_INPUT_ERROR:
            return "Error (bad input): " + self.msg
        elif self.flag == EXIT_TR_INCREASE_ERROR:
            return "Error (trust region increase): " + self.msg
        elif self.flag == EXIT_LINALG_ERROR:
            return "Error (linear algebra): " + self.msg
        else:
            return "Unknown exit flag: " + self.msg


# A container for the results of the optimization routine
class OptimResults(object):
    def __init__(self, xmin, fmin, nf, niter, exit_flag, exit_msg):
        self.x = xmin
        self.f = fmin
        self.nf = nf
        self.niter = niter
        self.flag = exit_flag
        self.msg = exit_msg

        # Set standard names for exit flags
        self.EXIT_MAXFUN_WARNING = EXIT_MAXFUN_WARNING
        self.EXIT_SUCCESS = EXIT_SUCCESS
        self.EXIT_INPUT_ERROR = EXIT_INPUT_ERROR
        self.EXIT_TR_INCREASE_ERROR = EXIT_TR_INCREASE_ERROR
        self.EXIT_LINALG_ERROR = EXIT_LINALG_ERROR

    def __str__(self):
        # Result of calling print(soln)
        output = "****** QARSTA Results ******\n"
        if self.flag != self.EXIT_INPUT_ERROR:
            output += "Solution xmin = %s\n" % str(self.x)
            output += "Objective value f(xmin) = %.10g\n" % self.f
            output += "Needed %g objective evaluations\n" % (self.nf)
            output += "Total iteration = %g\n" % self.niter
        output += "Exit flag = %g\n" % self.flag
        output += "%s\n" % self.msg
        output += "****************************\n"
        return output
