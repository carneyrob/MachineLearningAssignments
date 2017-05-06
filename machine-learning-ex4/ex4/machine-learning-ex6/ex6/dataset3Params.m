function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


testC = [ .01, .03, .1, .3, 1, 3, 10, 30];
testS = [ .01, .03, .1, .3, 1, 3, 10, 30];
best = 1;
bestC = 0;
bestS = 0;
for i = 1:8
    Cph = testC(i)
    for j = 1:8
        sigmaph = testS(j);
        model = svmTrain(X, y, Cph, @(x1, x2) gaussianKernel(x1, x2, sigmaph));
        yPred = svmPredict(model, Xval);
        error = mean(double(yPred ~= yval));
        if error < best
            best = error;
            bestC = Cph;
            bestS = sigmaph;
        end
    end
end
C = bestC; 
sigma = bestS;

% =========================================================================

end
