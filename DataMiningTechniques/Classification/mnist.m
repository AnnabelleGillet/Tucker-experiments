%% Load file and construct tensor
fid = fopen('datasets/mnist_784_csv.csv');
tline = fgetl(fid); % skip header
tline = fgetl(fid);
mnist_data = zeros(28, 28, 8000, 10);
classes_idx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
while ischar(tline)
    line = split(tline, ',');
    classe = str2num(line{28 * 28 + 1});
    classe_idx = classes_idx{classe + 1} + 1;
    classes_idx{classe + 1} = classes_idx{classe + 1} + 1;
    line_index = 1;
    for i = 1:28
        for j = 1:28
            mnist_data(i, j, classe_idx, classe + 1) = str2num(line{line_index});
            line_index = line_index + 1;
        end
    end
    tline = fgetl(fid);
end
fclose(fid);

%% Cross validation
% Split differently the train and test set to validate more precisely the
% quality of the classification. The metrics computed are the global
% precision (how many elements have been affected the correct class
% compared to the total number of elements), the precision per class (how
% many elements that belongs to the class have been affected to this
% class), the recall per class (how many elements that have been affected 
% to the class really belongs to this class), and the F1-score (the
% harmonic mean between precision and recall for this class).
nb_iterations = 5;
mnist_train_limit = 2000;
start_index_per_iteration = (6000 - mnist_train_limit) / nb_iterations;
hooi_mnist_eval = {};
mnist_global_precisions = zeros(nb_iterations, 1);
mnist_class_precisions = zeros(nb_iterations, 10);
mnist_class_recalls = zeros(nb_iterations, 10);
mnist_class_f1s = zeros(nb_iterations, 10);

for iteration = 1:nb_iterations
    fprintf('Iteration %2d\n', iteration);
    % Split data 
    start_index = 1 + ((iteration - 1) * start_index_per_iteration);
    mnist_train_tensor = tensor(mnist_data(:, :, start_index:start_index + mnist_train_limit - 1, :));
    mnist_test_data = zeros(28, 28, 8000 - mnist_train_limit, 10);
    if iteration > 1
        mnist_test_data(:, :, 1:start_index - 1, :) = mnist_data(:, :, 1:start_index - 1, :);
    end
    if iteration < nb_iterations
        mnist_test_data(:, :, start_index:8000 - mnist_train_limit, :) = mnist_data(:, :, start_index + mnist_train_limit:8000, :);
    end

    % Run HOOI decomposition
    mnist_hooi = tucker_als2(mnist_train_tensor, [9 8 1 10]);

    % Classification of test data
    hooi_mnist_eval_iteration = zeros(10, 10);
    for class = 1:10
        fprintf(' Digit %2d\n', class - 1);
        for sample = 1:(8000 - mnist_train_limit)
            if sum(sum(mnist_test_data(:, :, sample, class))) > 0
                data_to_classify = zeros(28, 28, mnist_train_limit);
                data_to_classify(:, :, :) = repmat(mnist_test_data(:, :, sample, class), 1, 1, mnist_train_limit);
                data_to_classify = tensor(data_to_classify, [28, 28, mnist_train_limit, 1]);
                data_to_classify = ttm(data_to_classify, mnist_hooi.U{1}', 1);
                data_to_classify = ttm(data_to_classify, mnist_hooi.U{2}', 2);
                data_to_classify = ttm(data_to_classify, mnist_hooi.U{3}', 3);
                min_n = realmax("double");
                best_class = 0;
                for class_to_try = 1:10
                    data_to_try = ttm(mnist_hooi.core, mnist_hooi.U{4}(class_to_try, :), 4);
                    n = norm(data_to_try - data_to_classify);
                    if n < min_n
                        best_class = class_to_try;
                        min_n = n;
                    end
                end
                hooi_mnist_eval_iteration(class, best_class) = hooi_mnist_eval_iteration(class, best_class) + 1;
            end
        end
    end
    hooi_mnist_eval{iteration} = hooi_mnist_eval_iteration;
    % Computation of metrics
    mnist_global_precisions(iteration) = trace(hooi_mnist_eval_iteration)/sum(sum(hooi_mnist_eval_iteration));
    for class = 1:10
        class_precision = hooi_mnist_eval_iteration(class, class) / sum(hooi_mnist_eval_iteration(:, class));
        class_recall = hooi_mnist_eval_iteration(class, class) / sum(hooi_mnist_eval_iteration(class, :));
        class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall);
        mnist_class_precisions(iteration, class) = class_precision;
        mnist_class_recalls(iteration, class) = class_recall;
        mnist_class_f1s(iteration, class) = class_f1;
    end
end

%% Display metrics
% Global precision
mnist_global_precisions
mean(mnist_global_precisions)
% Precision per class
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(mnist_class_precisions, 'Title', 'Class precision for the mnist dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(mnist_class_precisions), 'XLabel', 'Class', 'YLabel', 'Mean')
% Recall per class
figure
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(mnist_class_recalls, 'Title', 'Class recall for the mnist dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(mnist_class_recalls), 'XLabel', 'Class', 'YLabel', 'Mean')
% F1-score per class
figure
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(mnist_class_f1s, 'Title', 'Class F1-score for the mnist dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(mnist_class_f1s), 'XLabel', 'Class', 'YLabel', 'Mean')
