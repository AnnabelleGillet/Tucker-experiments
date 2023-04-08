%% Load dataset and build tensor
% The first dimension represents the characteristics of the flowers (in the
% order, the sepal length, the sepal width, the petal length and the petal
% width), the second dimension represents the samples (there are 50 samples
% by species), and the third dimension represents the species. 
iris_data = zeros(4, 50, 3);
iris_classes = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
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
    
    iris_data(1, sample, class_index) = sepal_length;
    iris_data(2, sample, class_index) = sepal_width; 
    iris_data(3, sample, class_index) = petal_length; 
    iris_data(4, sample, class_index) = petal_width;
    sample = sample + 1;
    if sample == 51
        sample = 1;
    end
    tline = fgetl(fid);
end

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
iris_train_limit = 20;
start_index_per_iteration = (50 - iris_train_limit) / nb_iterations;
hooi_iris_eval = {};
iris_global_precisions = zeros(nb_iterations, 1);
iris_class_precisions = zeros(nb_iterations, length(iris_classes));
iris_class_recalls = zeros(nb_iterations, length(iris_classes));
iris_class_f1s = zeros(nb_iterations, length(iris_classes));

for iteration = 1:nb_iterations
    fprintf('Iteration %2d\n', iteration);
    % Split data 
    start_index = 1 + ((iteration - 1) * start_index_per_iteration);
    iris_train_tensor = tensor(iris_data(:, start_index:start_index + iris_train_limit - 1, :));
    iris_test_data = zeros(4, 50 - iris_train_limit, 3);
    if iteration > 1
        iris_test_data(:, 1:start_index - 1, :) = iris_data(:, 1:start_index - 1, :);
    end
    if iteration < nb_iterations
        iris_test_data(:, start_index:50 - iris_train_limit, :) = iris_data(:, start_index + iris_train_limit:50, :);
    end

    % Run HOOI decomposition
    iris_hooi = tucker_als2(iris_train_tensor, [2 3 3]);

    % Classification of test data
    hooi_iris_eval_iteration = zeros(3, 3);
    for class = 1:3
        fprintf(' Iris %2d\n', class);
        for sample = 1:(50 - iris_train_limit)
            data_to_classify = zeros(4, iris_train_limit);
            data_to_classify(:, :) = repmat(iris_test_data(:, sample, class), 1, iris_train_limit);
            data_to_classify = tensor(data_to_classify, [4, iris_train_limit, 1]);
            data_to_classify = ttm(data_to_classify, iris_hooi.U{1}', 1);
            data_to_classify = ttm(data_to_classify, iris_hooi.U{2}', 2);
            min_n = realmax("double");
            best_class = 0;
            for class_to_try = 1:3
                data_to_try = ttm(iris_hooi.core, iris_hooi.U{3}(class_to_try, :), 3);
                n = norm(data_to_try - data_to_classify);
                if n < min_n
                    best_class = class_to_try;
                    min_n = n;
                end
            end
            hooi_iris_eval_iteration(class, best_class) = hooi_iris_eval_iteration(class, best_class) + 1;
        end
    end
    hooi_iris_eval{iteration} = hooi_iris_eval_iteration;
    % Computation of metrics
    iris_global_precisions(iteration) = trace(hooi_iris_eval_iteration)/sum(sum(hooi_iris_eval_iteration));
    for class = 1:3
        class_precision = hooi_iris_eval_iteration(class, class) / sum(hooi_iris_eval_iteration(:, class));
        class_recall = hooi_iris_eval_iteration(class, class) / sum(hooi_iris_eval_iteration(class, :));
        class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall);
        iris_class_precisions(iteration, class) = class_precision;
        iris_class_recalls(iteration, class) = class_recall;
        iris_class_f1s(iteration, class) = class_f1;
    end
end

%% Display metrics
% Global precision
iris_global_precisions
mean(iris_global_precisions)
% Precision per class
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(iris_class_precisions, 'Title', 'Class precision for the Iris dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(iris_class_precisions), 'XLabel', 'Class', 'YLabel', 'Mean')
% Recall per class
figure
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(iris_class_recalls, 'Title', 'Class recall for the Iris dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(iris_class_recalls), 'XLabel', 'Class', 'YLabel', 'Mean')
% F1-score per class
figure
t = tiledlayout(nb_iterations, 1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile([nb_iterations-1 1])
heatmap(iris_class_f1s, 'Title', 'Class F1-score for the Iris dataset', 'YLabel', 'Iteration')
nexttile
heatmap(mean(iris_class_f1s), 'XLabel', 'Class', 'YLabel', 'Mean')
