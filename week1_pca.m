% PCA
load('wine.mat');

figure
parallelplot(wine)

figure
glyphplot(wine)

x = normalize(wine(:,2:end));
y = wine(:,1);

[coeff, score, latent, ~, explained, mu] = pca(x);

figure
gscatter(score(:,1), score(:,2), y, 'rgb');
xlabel('PC1');
ylabel('PC2');

figure
bar(1:13, cumsum(latent)/sum(latent))
xlabel('Number of principal components') 
ylabel('Cumulative contribution rate') 
title('PCA Cumulative contribution rate')

figure
stairs(1:13, explained)
xlabel('Number of principal components') 
ylabel('Individual contribution rate') 
title('PCA Individual contribution rate')

% next homework
clear;
data = readmatrix('data.csv');
data(:,3:end);
parallelplot(data(:,3:end))
s = readtable('data.csv');
y = string(s{:,2});


x = normalize(data(:,3:end));

[coeff, score, latent, ~, explained, mu] = pca(x);
figure;
gscatter(score(:,1), score(:,2), y, 'rb');
xlabel('PC1');
ylabel('PC2');

figure
bar(1:30, cumsum(latent)/sum(latent))
xlabel('Number of principal components') 
ylabel('Cumulative contribution rate') 
title('PCA Cumulative contribution rate')