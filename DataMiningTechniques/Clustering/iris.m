%% Load dataset and build tensor
% The first dimension represents the characteristics of the flowers (in the
% order, the sepal length, the sepal width, the petal length and the petal
% width), the second dimension represents the samples (there are 50 samples
% by species, and 3 species). 
iris_data = zeros(4, 50 * 3);
iris_classes = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
iris_sample_classes = zeros(50 * 3, 1);
fid = fopen('datasets/iris.csv');
tline = fgetl(fid); 
sample = 1;
while ischar(tline)
    line = split(tline, ',');
    sepal_length = str2double(line{1});
    sepal_width = str2double(line{2});
    petal_length = str2double(line{3});
    petal_width = str2double(line{4});
    class_index = find(strcmp(iris_classes, line{5}));
    iris_sample_classes(sample) = class_index;
    
    iris_data(1, sample) = sepal_length;
    iris_data(2, sample) = sepal_width; 
    iris_data(3, sample) = petal_length; 
    iris_data(4, sample) = petal_width;
    sample = sample + 1;
    tline = fgetl(fid);
end
iris_tensor = tensor(iris_data);

%% Choose the best ranks
find_best_ranks(iris_tensor, .95);

%% Run HOOI decomposition
iris_hooi = tucker_als2(iris_tensor, [3, 3]);

%% Find clusters with k-medoids
hooi_iris_clusters = Kmedoids(iris_hooi.U{2}, 3);

%% Build confusion matrix
iris_eval = zeros(3, 3);
for i = 1:length(hooi_iris_clusters)
    class = iris_sample_classes(i);
    iris_eval(class, hooi_iris_clusters(i)) = iris_eval(class, hooi_iris_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:3
    [val, idx] = sort(iris_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > iris_eval(idx, idx)))
        tmp = iris_eval(:, i);
        iris_eval(:, i) = iris_eval(:, idx);
        iris_eval(:, idx) = tmp;
    end
end

%% Show result
hooi_precision = sum(diag(iris_eval)) / length(hooi_iris_clusters)
heatmap(iris_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')