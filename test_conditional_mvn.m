% test_conditional_mvn.m : test conditional_mvnpdf_low_rank
%   current only have bivaraite normal distribution test

mu = [1; -1];
M  = [0.9 0.9; 0.7 0.8];
d  = [0.5; 0.4];

% build the convrance from low ranl decomposition
sigma = M * M' + diag(d);

% get the conditional dist on y = 0.5 while x linearly spacing
x = (-1:0.01:8);
y_given = ones(1, n)*0.5;

[~, n] = size(x);
X = cat( 1, x, y_given )';
y = mvnpdf(X,mu',sigma');

% get the conditional Gaussain on the y2 = 0.5
y2  = ones(1, 1) * 0.5;
mu1 = ones(1, 1) * 1;
mu2 = ones(1, 1) * -1;
M1  = M(1, :);
M2  = M(2, :);
d1  = ones(1, 1) * 0.5;
d2  = ones(1, 1) * 0.4;

[mu1, Sigma11] = conditional_mvnpdf_low_rank(y2, mu1, mu2, M1, M2, d1, d2);

% argmax to get the peak position of the 1-d normal distr 
[argval, argmax] = max(y);
mean_pdf = x(argmax);

% should be the same as our conditional 1-d normal distr mu
assert( abs((mean_pdf - mu1)) < 0.02 )

% test if the shape of the conditional distr are the same
y_cond = mvnpdf(x', ones(1, 1) * mu1, ones(1, 1) * Sigma11);

% normalise both of them
y_cond_norm = y_cond / sum(y_cond);
y_norm      = y / sum(y);

assert( all( (y_cond_norm - y_norm) < 1e-4 ) )
