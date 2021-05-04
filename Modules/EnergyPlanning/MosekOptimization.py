import mosek
from numpy import ndarray
import sys
import numpy as np


class MosekProcess(object):

    def __init__(self, Aeq: ndarray, Beq: ndarray, A: ndarray, B: ndarray, c: ndarray, lb: ndarray, ub: ndarray):

        self.Aeq = np.round(Aeq, 7)
        self.Beq = np.round(Beq, 7)
        self.A = np.round(A, 7)
        self.B = np.round(B, 7)
        self.c = np.round(c, 7)
        self.lb = np.round(lb, 7)
        self.ub = np.round(ub, 7)

        # Condition to use only in Mosek Process
        # Aeq_new = np.zeros((1, Aeq.shape[1]))
        # Aeq_new[0, -1] = -1.0
        # self.Aeq = np.concatenate((np.round(Aeq, 7), Aeq_new), axis=0)
        # Beq_new = np.zeros((1, 1))
        # self.Beq = np.concatenate((np.round(Beq, 7), Beq_new))

        # Condition to use only in Mosek Process
        # A_new = np.zeros((1, A.shape[1]))
        # A_new[0, -1] = -1.0
        # self.A = np.concatenate((np.round(A, 7), A_new), axis=0)
        # B_new = np.zeros((1, 1))
        # self.B = np.concatenate((np.round(B, 7), B_new))

    # Define a stream printer to grab output from MOSEK
    @staticmethod
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def run(self) -> tuple:

        inf = 0.0

        # Make mosek environment
        with mosek.Env() as env:
            # Create a task object
            with env.Task(0, 0) as task:

                # Attach a log stream printer to the task
                # task.set_Stream(mosek.streamtype.log, self.streamprinter)

                # Bound keys and values for variables
                bkx = [0.] * len(self.c)
                blx = [0.] * len(self.c)
                bux = [0.] * len(self.c)
                for ivar in range(len(self.c)):
                    if np.isnan(self.lb[ivar]) and np.isnan(self.ub[ivar]):
                        bkx[ivar] = mosek.boundkey.fr
                        blx[ivar] = -inf
                        bux[ivar] = +inf
                    elif self.lb[ivar] < self.ub[ivar]:
                        bkx[ivar] = mosek.boundkey.ra
                        blx[ivar] = self.lb[ivar]
                        bux[ivar] = self.ub[ivar]
                    elif (self.lb[ivar] == self.ub[ivar]) and not np.isnan(self.lb[ivar]):
                        bkx[ivar] = mosek.boundkey.fx
                        blx[ivar] = self.lb[ivar]
                        bux[ivar] = self.ub[ivar]
                    else:
                        bkx[ivar] = mosek.boundkey.lo
                        blx[ivar] = self.lb[ivar]
                        bux[ivar] = +inf

                # Below is the sparse representation of the A matrix stored by column.
                asub = list()
                aval = list()
                for icol in range(len(self.c)):
                    aux1 = list()
                    aux2 = list()
                    for ilin in range(self.Aeq.shape[0]):
                        if self.Aeq[ilin, icol] != 0.:
                            aux1.append(ilin)
                            aux2.append(self.Aeq[ilin, icol])
                    if self.A.shape[0] != 0:
                        for ilin in range(self.A.shape[0]):
                            if self.A[ilin, icol] != 0.:
                                aux1.append(self.Aeq.shape[0] + ilin)
                                aux2.append(self.A[ilin, icol])
                    asub.append(aux1)
                    aval.append(aux2)

                # Bound keys and values for constraints
                bkc = [0.] * (self.Aeq.shape[0] + self.A.shape[0])
                blc = [0.] * (self.Aeq.shape[0] + self.A.shape[0])
                buc = [0.] * (self.Aeq.shape[0] + self.A.shape[0])
                for ilin in range(self.Aeq.shape[0]):
                    bkc[ilin] = mosek.boundkey.fx
                    blc[ilin] = self.Beq[ilin, 0]
                    buc[ilin] = self.Beq[ilin, 0]
                if self.A.shape[0] != 0:
                    for ilin in range(self.A.shape[0]):
                        bkc[self.Aeq.shape[0] + ilin] = mosek.boundkey.up
                        blc[self.Aeq.shape[0] + ilin] = -inf
                        buc[self.Aeq.shape[0] + ilin] = self.B[ilin]

                numvar = len(bkx)
                numcon = len(bkc)

                # Append 'numcon' empty constraints.
                # The constraints will initially have no bounds.
                task.appendcons(numcon)

                # Append 'numvar' variables.
                # The variables will initially be fixed at zero (x=0).
                task.appendvars(numvar)

                for i in range(numvar):
                    # Set the linear term c_j in the objective.
                    task.putcj(i, self.c[i])

                    # Set the bounds on variable j
                    # blx[j] <= x_j <= bux[j]
                    task.putvarbound(i, bkx[i], blx[i], bux[i])

                # Input column j of A
                for i in range(len(asub)):
                    task.putacol(i, asub[i], aval[i])

                # Set the bounds on constraints.
                # blc[i] <= constraint_i <= buc[i]
                for j in range(numcon):
                    task.putconbound(j, bkc[j], blc[j], buc[j])

                # Input the objective sense (minimize/maximize)
                task.putobjsense(mosek.objsense.minimize)

                # Define variables to be integers
                # task.putvartypelist(var_integer_list, [mosek.variabletype.type_int]*len(var_integer_list))

                # Solve the problem
                # task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
                task.optimize()

                # Print a summary containing information
                # about the solution for debugging purposes
                # task.solutionsummary(mosek.streamtype.msg)

                # Get status information about the solution
                solsta = task.getsolsta(mosek.soltype.bas)
                # solsta = task.getsolsta(mosek.soltype.itg)

                x, dual, fval = 0, 0, 0
                if solsta == mosek.solsta.optimal:

                    x = [0.] * numvar
                    dual = [0.] * numcon
                    task.getxx(mosek.soltype.bas, x)
                    task.gety(mosek.soltype.bas, dual)
                    fval = task.getprimalobj(mosek.soltype.bas)

                    # task.getxx(mosek.soltype.itg, x)
                    # task.gety(mosek.soltype.itg, dual)
                    # fval = task.getprimalobj(mosek.soltype.itg)

                    # print("Optimal solution: ")
                    # for i in range(numvar):
                    #     print("x[" + str(i) + "]=" + str(xx[i]))

                elif solsta == mosek.solsta.dual_infeas_cer or solsta == mosek.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility certificate found.\n")

                elif solsta == mosek.solsta.unknown:
                    print("Unknown solution status")

                else:
                    print("Other solution status")

        return x, dual, fval
