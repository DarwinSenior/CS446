% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here

X = data(:,1:n);
Y = data(:, n+1);

A = [[([X,ones(m,1)].*repmat(Y, 1, n+1))',zeros(n+1, 1)]',ones(m+1, 1)];

c = [zeros(1,n+1),1]';

b = [ones(1,n+1),0]';

%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end
