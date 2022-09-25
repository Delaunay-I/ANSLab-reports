#importing the necessary libraries
import numpy as np
import sys

# total arguments
n = len(sys.argv)
if n<3:
    sys.exit("\nWrong number of arguments?")
print("\nTotal arguments passed:", n)
 
# Arguments passed
print("Name of Python script:", sys.argv[0])

sFiletoRead = sys.argv[1]
sFileName = sys.argv[2]

#reading the Jacobian Matrix file, which is in CSV format exported from MATLAB
#JacMat = np.genfromtxt(CSVData, delimiter=",")
JacMat = np.empty([7098, 7098])

def read_mpibaij(file):
    lines = file.read().splitlines()
    assert 'Mat Object:' in lines[0]
    assert lines[1] == '  type: mpibaij'
    for line in lines[2:]:
        parts = line.split(': ')
        assert len(parts) == 2
        assert parts[0].startswith('row ')

        row_index = int(parts[0][4:])
        row_contents = eval(parts[1].replace(')  (', '), ('))

        # Here you have the row_index and a tuple of (column_index, value)
        # pairs that specify the non-zero contents. You could process this
        # depending on your needs, e.g. store the values in an array.
        for (col_index, value) in row_contents:
            #print('row %d, col %d: %s' % (row_index, col_index, value))
            # TODO: Implement real code here.
            # You probably want to do something like:
            JacMat[row_index][col_index] = value

with open(sFiletoRead, 'rt', encoding='ascii') as file:
        read_mpibaij(file)
        
#Solving the eigen system with the lapack package in numpy
eigenvalues, eigenvectors = np.linalg.eig(JacMat)

#pack eigenvectors and eigenvalues together, and sort them
#so that we can get eigenvectors is the order of importance
eigenpair = tuple(zip(eigenvalues, eigenvectors))
sortedEigenpair = sorted(eigenpair, key=lambda tup: tup[0], reverse=True)
numpyeigenPair = np.array(sortedEigenpair)
#print("An eigenpair is produced.\n1st element type:",type(numpyeigenPair[1, 0]), " 2nd element type:", type(numpyeigenPair[1,1]))


#extrac the first elements of array which are the eigenvalues
eigvals = numpyeigenPair[:, 0] 
#extract the second elements of array which are the eigenvectors
eigvecs = numpyeigenPair[:, 1] 

#defining the function to write the eigenvalues to a .dat file
def printEigenvaluesDAT(sFileName, eigenarray):
    Rel = [ele.real for ele in eigenarray]
    Imag = [ele.imag for ele in eigenarray]
    
    with open(sFileName + "_eigenvalues.dat", "w") as file:
        file.write("VARIABLES = \"Real\" \"Imag\"\n")
        for x in zip(Rel, Imag):
            file.write("{0}\t{1}\n".format(*x))
# Write all the eigenvalues    
printEigenvaluesDAT(sFileName, eigvals)
