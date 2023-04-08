%% Load file and construct tensor
% 1000 samples par class are kept. The first two dimensions represent the
% pixels, and the last one represents the samples. 
fid = fopen('datasets/mnist_784_csv.csv');
tline = fgetl(fid);
tline = fgetl(fid);
classes = zeros(10000, 1);
mnist_data = zeros(28, 28, 10000);
samples_per_class = zeros(10, 1);
sample = 1;
while ischar(tline)
    line = split(tline, ',');
    class = str2num(line{28 * 28 + 1});
    if samples_per_class(class + 1) < 1000
        line_index = 1;
        for i = 1:28
            for j = 1:28
                mnist_data(i, j, sample) = str2num(line{line_index});
                line_index = line_index + 1;
            end
        end
        classes(sample) = class;
        sample = sample + 1;
        samples_per_class(class + 1) = samples_per_class(class + 1) + 1;
    end
    tline = fgetl(fid);
end
fclose(fid);
mnist_tensor = tensor(mnist_data);

%% Find best ranks
find_best_ranks(mnist_tensor, .95);

%% Run HOOI decomposition
mnist_hooi = tucker_als2(mnist_tensor, [10, 10, 100]);

%% Find clusters with k-medoids
hooi_mnist_clusters = Kmedoids(mnist_hooi.U{3}, 10);

%% Build confusion matrix
mnist_eval = zeros(10, 10);
for i = 1:length(hooi_mnist_clusters)
    mnist_eval(classes(i) + 1, hooi_mnist_clusters(i)) = mnist_eval(classes(i) + 1, hooi_mnist_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:100
    [val, idx] = sort(mnist_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > mnist_eval(idx, idx)))
        tmp = mnist_eval(:, i);
        mnist_eval(:, i) = mnist_eval(:, idx);
        mnist_eval(:, idx) = tmp;
    end
end

%% Show result
hooi_precision = sum(diag(mnist_eval)) / length(hooi_mnist_clusters)
heatmap(mnist_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')