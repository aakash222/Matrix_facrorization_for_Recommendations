import sys
import numpy as np
from sklearn.model_selection import train_test_split

fin =  sys.argv[1];
fout = sys.argv[2];

def _split(a):
	count = 0;
	l1 = [0]*610;
	test = list();
	train = list();
	for i in range(len(a)):
		l1[int(a[i][0])] += 1;
	for i in range(len(a)):
		if l1[int(a[i][0])]>2 and count<200:
			l1[int(a[i][0])]=0;
			count += 1;
			test.append(a[i]);
		else:
			train.append(a[i]);
	return train,test;

def matrix_factorization(R, P, Q, K, steps, alpha, beta):
	Q = Q.T
	for step in range(steps):
		print('step',step);        
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - np.dot(P[i,:],Q[:,j])
					for k in range(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		
	return P, Q.T
	

f_in = open(fin,"r");
f_in0 = f_in.readlines()[1:];
a = list();
for i in f_in0:
	a.append(list(map(float,i.split())));
f_in.close();

row = -1;
col = -1;
steps = 20;
alpha = 0.002;
beta = 0.02;
dim = int(20);
for i in range(len(a)):
	if row < a[i][0]:
		row = a[i][0];
	if col < a[i][1]:
		col = a[i][1];
	
row=int(row+1);
col=int(col+1);

U = np.random.rand(row, dim);
V = np.random.rand(col, dim);



in_matrix = [[0.0]*col for i in range(row)];
X_train, X_test = _split(a);


for i in range(len(X_train)):
	in_matrix[int(X_train[i][0])][int(X_train[i][1])] = X_train[i][2];

nU, nV = matrix_factorization(in_matrix,U,V,dim,steps,alpha,beta);

print("factorization done!!!");

res = np.matmul(nU, nV.T);
np.around(res,6);
np.savetxt(fout,res,fmt = '%f');
e = 0;
for i in range(len(X_test)):
	temp = res[int(X_test[i][0])][int(X_test[i][1])];
	e += pow(temp - X_test[i][2],2);
e /= len(X_test);
print('error ',e**0.5);
