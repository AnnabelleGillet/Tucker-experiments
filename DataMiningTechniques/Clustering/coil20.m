%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% With position %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load images and construct tensor
% The first two dimensions of the tensor represent the pixels, the
% third dimension represents the positions and the fourth the samples.
coil_data = zeros(128, 128, 72, 20 * 72);
coil_sample_objects = zeros(20 * 72, 1);
coil_sample_positions = zeros(20 * 72, 1);
sample = 1;
for object = 1:20
    for position = 0:71
        coil_data(:, :, position + 1, sample) = imread(strcat('datasets/coil/coil-20-proc/obj',num2str(object),'__',num2str(position),'.png'));
        coil_sample_objects(sample) = object;
        coil_sample_positions(sample) = position + 1;
        sample = sample + 1;
    end
end
coil_tensor = tensor(coil_data);

%% Compute ranks
find_best_ranks(coil_tensor, .99);

%% Run HOOI decomposition
coil_hooi = tucker_als2(coil_tensor, [31, 18, 72, 20]);

%% Find clusters with k-medoids
hooi_coil_clusters = Kmedoids(coil_hooi.U{4}, 20);

%% Build confusion matrix for objects
coil_eval = zeros(20, 20);
for i = 1:length(hooi_coil_clusters)
    class = coil_sample_objects(i);
    coil_eval(class, hooi_coil_clusters(i)) = coil_eval(class, hooi_coil_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:20
    [val, idx] = sort(coil_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > coil_eval(idx, idx)))
        tmp = coil_eval(:, i);
        coil_eval(:, i) = coil_eval(:, idx);
        coil_eval(:, idx) = tmp;
    end
end

%% Show result for objects
hooi_precision = sum(diag(coil_eval)) / length(hooi_coil_clusters)
heatmap(coil_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')

%% Rand Index and Adjusted Rand Index for objects
coil_ri = rand_index(hooi_coil_clusters, coil_sample_objects);
coil_ri
coil_ari = adjusted_rand_index(coil_eval);
coil_ari

%% Run HOOI decomposition
coil_hooi = tucker_als2(coil_tensor, [31, 18, 72, 72]);

%% Find clusters with k-medoids for positions
hooi_coil_clusters = Kmedoids(coil_hooi.U{4}, 72);

%% Build confusion matrix for positions
coil_eval = zeros(72, 72);
for i = 1:length(hooi_coil_clusters)
    class = floor((i - 1) / 20) + 1;
    coil_eval(class, hooi_coil_clusters(i)) = coil_eval(class, hooi_coil_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:72
    [val, idx] = sort(coil_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > coil_eval(idx, idx)))
        tmp = coil_eval(:, i);
        coil_eval(:, i) = coil_eval(:, idx);
        coil_eval(:, idx) = tmp;
    end
end

%% Show result for positions
hooi_precision = sum(diag(coil_eval)) / length(hooi_coil_clusters)
heatmap(coil_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')

%% Rand Index and Adjusted Rand Index for positions
coil_ri = rand_index(hooi_coil_clusters, coil_sample_positions);
coil_ri
coil_ari = adjusted_rand_index(coil_eval);
coil_ari

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Without position %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load images and construct tensor
% The first two dimensions of the tensor represent the pixels, the
% third dimension represents the samples.
coil_data = zeros(128, 128, 20 * 72);
coil_sample_objects = zeros(20 * 72, 1);
coil_sample_positions = zeros(20 * 72, 1);
sample = 1;
for object = 1:20
    for position = 0:71
        coil_data(:, :, sample) = imread(strcat('datasets/coil/coil-20-proc/obj',num2str(object),'__',num2str(position),'.png'));
        coil_sample_objects(sample) = object;
        coil_sample_positions(sample) = position + 1;
        sample = sample + 1;
    end
end
coil_tensor = tensor(coil_data);

%% Compute ranks
find_best_ranks(coil_tensor, .99);

%% Run HOOI decomposition for objects
coil_hooi = tucker_als2(coil_tensor, [31, 18, 20]);

%% Find clusters with k-medoids for objects
hooi_coil_clusters = Kmedoids(coil_hooi.U{3}, 20);

%% Build confusion matrix for objects
coil_eval = zeros(20, 20);
for i = 1:length(hooi_coil_clusters)
    class = floor((i - 1) / 72) + 1;
    coil_eval(class, hooi_coil_clusters(i)) = coil_eval(class, hooi_coil_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:20
    [val, idx] = sort(coil_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > coil_eval(idx, idx)))
        tmp = coil_eval(:, i);
        coil_eval(:, i) = coil_eval(:, idx);
        coil_eval(:, idx) = tmp;
    end
end

%% Show result for objects
hooi_precision = sum(diag(coil_eval)) / length(hooi_coil_clusters)
heatmap(coil_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')

%% Rand Index and Adjusted Rand Index for objects
coil_ri = rand_index(hooi_coil_clusters, coil_sample_objects);
coil_ri
coil_ari = adjusted_rand_index(coil_eval);
coil_ari

%% Run HOOI decomposition for positions
coil_hooi = tucker_als2(coil_tensor, [31, 18, 72]);

%% Find clusters with k-medoids for positions
hooi_coil_clusters = Kmedoids(coil_hooi.U{3}, 72);

%% Build confusion matrix for positions
coil_eval = zeros(72, 72);
for i = 1:length(hooi_coil_clusters)
    class = mod(i - 1, 72) + 1;
    coil_eval(class, hooi_coil_clusters(i)) = coil_eval(class, hooi_coil_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:72
    [val, idx] = sort(coil_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > coil_eval(idx, idx)))
        tmp = coil_eval(:, i);
        coil_eval(:, i) = coil_eval(:, idx);
        coil_eval(:, idx) = tmp;
    end
end

%% Show result for positions
hooi_precision = sum(diag(coil_eval)) / length(hooi_coil_clusters)
heatmap(coil_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')

%% Rand Index and Adjusted Rand Index for positions
coil_ri = rand_index(hooi_coil_clusters, coil_sample_positions);
coil_ri
coil_ari = adjusted_rand_index(coil_eval);
coil_ari