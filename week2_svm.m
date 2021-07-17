% SVM classifiation
load('wine.mat');
indices = 1:130;
X = wine(indices,2:3);
Y = wine(indices,1);

figure
gscatter(X(:,1), X(:,2), Y, 'rgb');
xlabel('Alcohol')
ylabel('Malic acid')


% liner SVM, C=1, soft margin
SVMModel = fitcsvm(X,Y, "BoxConstraint", 1);
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('Alcohol')
ylabel('Malic acid')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off

% C=1000, hard margin
SVMModel = fitcsvm(X,Y, "BoxConstraint", 1000);
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('Alcohol')
ylabel('Malic acid')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter", "latex");
hold off

% RBF kernel, non-linear
SVMModel = fitcsvm(X,Y, "BoxConstraint", 1, "KernelFunction", "gaussian");
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('Alcohol')
ylabel('Malic acid')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off

% multi-class, not 2 class
clear;
load('wine.mat');
X = wine(:,2:3);
Y = wine(:,1);

SVMModel = fitcecoc(X, Y, 'FitPosterior',true);
SVMModel.predict(X(1,:));
gscatter(X(:,1), X(:,2), Y, 'rgb');
hold on;
h=0.1;
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); 
[~,~,~, posteriors]  = SVMModel.predict([X1(:), X2(:)]);  % calculate scores at the each grid point
scatter(X1(:), X2(:) , 20, posteriors, "filled", "MarkerFaceAlpha", 0.2);
xlabel('Alcohol')
ylabel('Malic acid')
hold off;


% reduction of dimension by PCA->SVM
x = normalize(wine(:,2:end));
y = wine(:,1);
[coeff, score, latent, ~, explained, mu] = pca(x);
score;
figure;
gscatter(score(:,1), score(:,2), y, 'rgb');
xlabel('PC1');
ylabel('PC2');

% 2 class classification
indices = 1:130;
X = score(indices,1:2);
Y = y(indices,1);

SVMModel = fitcsvm(X,Y, "BoxConstraint", 1);
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score_] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score_(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('PC1')
ylabel('PC2')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off

% multi-class
X = score(:,1:2);
Y = y(:,1);
SVMModel = fitcecoc(X, Y, 'FitPosterior',true);
SVMModel.predict(X(1,:));
gscatter(X(:,1), X(:,2), Y, 'rgb');
hold on;
h=0.1;
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); 
[~,~,~, posteriors]  = SVMModel.predict([X1(:), X2(:)]);  % calculate scores at the each grid point
scatter(X1(:), X2(:) , 20, posteriors, "filled", "MarkerFaceAlpha", 0.2);
xlabel('PC1')
ylabel('PC2')


% https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
% download 'data.csv' from this url 
clear;
data = readmatrix('data.csv');
data(:,3:end);
X = normalize(data(:,3:end));
s = readtable('data.csv');
Y = string(s{:,2});
[coeff, score, latent, ~, explained, mu] = pca(X);
figure;
gscatter(score(:,1), score(:,2), Y, 'rg');
xlabel('PC1');
ylabel('PC2');
hold off;

% C=1, soft margin
X = score(:,1:2);
SVMModel = fitcsvm(X,Y, "BoxConstraint", 1);
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('PC1')
ylabel('PC2')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off

% RBF kernel, non-linear
SVMModel = fitcsvm(X,Y, "BoxConstraint", 1, "KernelFunction", "gaussian");
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('PC1')
ylabel('PC2')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off


% hyperparameter tuning, grid-search
SVMModel = fitcsvm(X, Y, "KernelFunction", "gaussian", ...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus', "Optimizer", "gridsearch"));


% plot result of grid-search
svInd = SVMModel.IsSupportVector; 
h = 0.1; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h:max(X(:,1)), min(X(:,2)):h:max(X(:,2))); % calculate grid points
[~,score] = SVMModel.predict([X1(:),X2(:)]);  % calculate scores at the each grid point
scoreGrid = reshape(score(:,2),size(X1,1),size(X2,2)); 
figure
gscatter(X(:,1), X(:,2), Y, 'rg');
hold on
plot(X(svInd,1),X(svInd,2),'ko','MarkerSize',10, "LineWidth",1) % Mark support vectors
contour(X1,X2,scoreGrid); % draw contour 
contour(X1,X2,scoreGrid, [0 0],'k', "LineWidth", 1); %draw decision bounday
contour(X1,X2,scoreGrid, [1 1],'k'); %draw f(x)=1
contour(X1,X2,scoreGrid, [-1 -1],'k'); %draw f(x)=-1
colorbar;
xlabel('PC1')
ylabel('PC2')
legend(["class1", "class2", "support vectors", "contour", "decision boundary($f(x)=0$)", "$f(x)=1$", "$f(x)=-1$"], "Interpreter","latex");
hold off