%% Load images and construct tensor
% The first two dimensions of the tensor represent the pixels, the
% third dimension represents the positions and the fourth the objects.
coil_data = zeros(128, 128, 72, 20);
for object = 1:20
    for img = 0:71
        coil_data(:, :, img + 1, object) = imread(strcat('datasets/coil/coil-20-proc/obj',num2str(object),'__',num2str(img),'.png'));
    end
end

%% Cross validation - without position
% Split differently the train and test set to validate more precisely the
% quality of the classification. The metrics computed are the global
% precision (how many elements have been affected the correct class
% compared to the total number of elements), the precision per class (how
% many elements that belongs to the class have been affected to this
% class), the recall per class (how many elements that have been affected 
% to the class really belongs to this class), and the F1-score (the
% harmonic mean between precision and recall for this class).
nb_iterations = 5;
nb_iterations_for_missing_values = 10;
coil_train_limit = 40;
start_index_per_iteration = (72 - coil_train_limit) / nb_iterations;
hooi_coil_eval = {};
coil_global_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 1);
coil_class_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 20);
coil_class_recalls = zeros(nb_iterations_for_missing_values, nb_iterations, 20);
coil_class_f1s = zeros(nb_iterations_for_missing_values, nb_iterations, 20);

for iteration = 1:nb_iterations
    fprintf('Iteration %2d\n', iteration);
    % Split data 
    start_index = floor(1 + ((iteration - 1) * start_index_per_iteration));
    coil_train_tensor = tensor(coil_data);
    coil_test_data = {}; % Object, position, image
    for i = 1:20
        start_index_object = i * start_index;
        while start_index_object + coil_train_limit > 72
            start_index_object = start_index_object + coil_train_limit - 72;
        end
        position_index = start_index_object;
        position_indexes = zeros(72 - coil_train_limit, 1);
        j = 1;
        for img = start_index_object:start_index_object + (72 - coil_train_limit) - 1
            position_indexes(j) = position_index;
            j = j + 1;
            image_to_test = coil_data(:, :, position_index, i);
            coil_test_data{length(coil_test_data)+1} = {i, position_index, image_to_test};
            position_index = position_index + 3;
            if position_index > 72
                position_index = position_index - 72;
            end
        end
        coil_train_tensor(:, :, position_indexes, i) = 0.0;
    end

    % Run HOOI decomposition
    coil_hooi = tucker_als2(coil_train_tensor, [20 20 72 20]);

    % Classification of test data
    hooi_coil_eval_iteration = zeros(nb_iterations_for_missing_values, 20, 20);
    initial_size = zeros(length(coil_test_data), 1);
    for iteration_missing_values = 1:nb_iterations_for_missing_values
        for i = 1:length(coil_test_data)
            class = coil_test_data{i}{1};
            position = coil_test_data{i}{2};
            % Remove data
            if iteration_missing_values == 1
                initial_size(i) = length(find(tensor(coil_test_data{i}{3})));
            else
                indexes = find(tensor(coil_test_data{i}{3}));
                number_of_elements_to_remove = floor(0.1 * initial_size(i));
                selected_indexes = indexes(randperm(length(indexes), number_of_elements_to_remove), :);
                tmp = tensor(coil_test_data{i}{3});
                tmp(selected_indexes) = 0.0;
                coil_test_data{i}{3} = tmp.data;
            end
            %
            data_to_classify = zeros(128, 128, 72);
            data_to_classify(:, :, :) = repmat(coil_test_data{i}{3}, 1, 1, 72);
            data_to_classify = tensor(data_to_classify, [128, 128, 72, 1]);
            data_to_classify = ttm(data_to_classify, coil_hooi.U{1}', 1);
            data_to_classify = ttm(data_to_classify, coil_hooi.U{2}', 2);
            data_to_classify = ttm(data_to_classify, coil_hooi.U{3}', 3);
            min_n = realmax("double");
            best_class = 0;
            best_position = 0;
            for class_to_try = 1:20
                data_to_try = ttm(coil_hooi.core, coil_hooi.U{4}(class_to_try, :), 4);
                n = norm(data_to_try - data_to_classify);
                if n < min_n
                    best_class = class_to_try;
                    min_n = n;
                end
            end
            hooi_coil_eval_iteration(iteration_missing_values, class, best_class) = hooi_coil_eval_iteration(iteration_missing_values, class, best_class) + 1;
        end
        % Computation of metrics
        coil_global_precisions(iteration_missing_values, iteration) = trace(squeeze(hooi_coil_eval_iteration(iteration_missing_values, :, :))) / sum(sum(squeeze(hooi_coil_eval_iteration(iteration_missing_values, :, :))));
        for class = 1:20
            class_precision = hooi_coil_eval_iteration(iteration_missing_values, class, class) / sum(hooi_coil_eval_iteration(iteration_missing_values, :, class));
            class_recall = hooi_coil_eval_iteration(iteration_missing_values, class, class) / sum(hooi_coil_eval_iteration(iteration_missing_values, class, :));
            class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall);
            coil_class_precisions(iteration_missing_values, iteration, class) = class_precision;
            coil_class_recalls(iteration_missing_values, iteration, class) = class_recall;
            coil_class_f1s(iteration_missing_values, iteration, class) = class_f1;
        end
    end
    hooi_coil_eval{iteration} = hooi_coil_eval_iteration;
end

%% Display metrics - without position
% Global precision
coil_global_precisions
mean(coil_global_precisions')
%% Precision per class
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
display_tile = 11;
for m = 1:nb_iterations_for_missing_values
    nexttile
    precision_matrix = squeeze(coil_class_precisions(m, :, :));
    precision_matrix(isnan(precision_matrix)) = 0;
    heatmap([squeeze(precision_matrix); NaN(1, 20); mean(squeeze(precision_matrix))], 'Title', strcat('Class precision for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% Recall per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    recall_matrix = squeeze(coil_class_recalls(m, :, :));
    recall_matrix(isnan(recall_matrix)) = 0;
    heatmap([squeeze(recall_matrix); NaN(1, 20); mean(squeeze(recall_matrix))], 'Title', strcat('Class recall for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% F1-score per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    f1_matrix = squeeze(coil_class_f1s(m, :, :));
    f1_matrix(isnan(f1_matrix)) = 0;
    heatmap([squeeze(f1_matrix); NaN(1, 20); mean(squeeze(f1_matrix))], 'Title', strcat('Class F1-score for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end

%% Cross validation - with position
% Split differently the train and test set to validate more precisely the
% quality of the classification. The metrics computed are the global
% precision (how many elements have been affected the correct class
% compared to the total number of elements), the precision per class (how
% many elements that belongs to the class have been affected to this
% class), the recall per class (how many elements that have been affected 
% to the class really belongs to this class), and the F1-score (the
% harmonic mean between precision and recall for this class).
nb_iterations = 5;
nb_iterations_for_missing_values = 10;
coil_train_limit = 40;
start_index_per_iteration = (72 - coil_train_limit) / nb_iterations;
hooi_coil_eval_object = {};
hooi_coil_eval_position = {};
coil_global_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 1);
coil_class_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 20);
coil_class_recalls = zeros(nb_iterations_for_missing_values, nb_iterations, 20);
coil_class_f1s = zeros(nb_iterations_for_missing_values, nb_iterations, 20);

for iteration = 1:nb_iterations
    fprintf('Iteration %2d\n', iteration);
    % Split data 
    start_index = floor(1 + ((iteration - 1) * start_index_per_iteration));
    coil_train_tensor = tensor(coil_data);
    coil_test_data = {}; % Object, position, image
    for i = 1:20
        start_index_object = i * start_index;
        while start_index_object + coil_train_limit > 72
            start_index_object = start_index_object + coil_train_limit - 72;
        end
        position_index = start_index_object;
        position_indexes = zeros(72 - coil_train_limit, 1);
        j = 1;
        for img = start_index_object:start_index_object + (72 - coil_train_limit) - 1
            position_indexes(j) = position_index;
            j = j + 1;
            image_to_test = coil_data(:, :, position_index, i);
            coil_test_data{length(coil_test_data)+1} = {i, position_index, image_to_test};
            position_index = position_index + 3;
            if position_index > 72
                position_index = position_index - 72;
            end
        end
        coil_train_tensor(:, :, position_indexes, i) = 0.0;
    end

    % Run HOOI decomposition
    coil_hooi = tucker_als2(coil_train_tensor, [20 20 72 20]);

    % Classification of test data
    hooi_coil_eval_object_iteration = zeros(nb_iterations_for_missing_values, 20, 20);
    hooi_coil_eval_position_iteration = zeros(nb_iterations_for_missing_values, 72, 72);
    initial_size = zeros(length(coil_test_data), 1);
    for iteration_missing_values = 1:nb_iterations_for_missing_values
        for i = 1:length(coil_test_data)
            class = coil_test_data{i}{1};
            position = coil_test_data{i}{2};
            % Remove data
            if iteration_missing_values == 1
                initial_size(i) = length(find(tensor(coil_test_data{i}{3})));
            else
                indexes = find(tensor(coil_test_data{i}{3}));
                number_of_elements_to_remove = floor(0.1 * initial_size(i));
                selected_indexes = indexes(randperm(length(indexes), number_of_elements_to_remove), :);
                tmp = tensor(coil_test_data{i}{3});
                tmp(selected_indexes) = 0.0;
                coil_test_data{i}{3} = tmp.data;
            end
            %
            data_to_classify = coil_test_data{i}{3};
            data_to_classify = tensor(data_to_classify, [128, 128, 1, 1]);
            data_to_classify = ttm(data_to_classify, coil_hooi.U{1}', 1);
            data_to_classify = ttm(data_to_classify, coil_hooi.U{2}', 2);
            min_n = realmax("double");
            best_class = 0;
            best_position = 0;
            for class_to_try = 1:20
                for position_to_try = 1:72
                    data_to_try = ttm(ttm(coil_hooi.core, coil_hooi.U{3}(position_to_try, :), 3), coil_hooi.U{4}(class_to_try, :), 4);
                    n = norm(data_to_try - data_to_classify);
                    if n < min_n
                        best_class = class_to_try;
                        best_position = position_to_try;
                        min_n = n;
                    end
                end
            end
            hooi_coil_eval_object_iteration(iteration_missing_values, class, best_class) = hooi_coil_eval_object_iteration(iteration_missing_values, class, best_class) + 1;
            hooi_coil_eval_position_iteration(iteration_missing_values, position, best_position) = hooi_coil_eval_position_iteration(iteration_missing_values, position, best_position) + 1;
        end
        % Computation of metrics
        coil_global_precisions(iteration_missing_values, iteration) = trace(squeeze(hooi_coil_eval_object_iteration(iteration_missing_values, :, :))) / sum(sum(squeeze(hooi_coil_eval_object_iteration(iteration_missing_values, :, :))))
        for class = 1:20
            class_precision = hooi_coil_eval_object_iteration(iteration_missing_values, class, class) / sum(hooi_coil_eval_object_iteration(iteration_missing_values, :, class));
            class_recall = hooi_coil_eval_object_iteration(iteration_missing_values, class, class) / sum(hooi_coil_eval_object_iteration(iteration_missing_values, class, :));
            class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall);
            coil_class_precisions(iteration_missing_values, iteration, class) = class_precision;
            coil_class_recalls(iteration_missing_values, iteration, class) = class_recall;
            coil_class_f1s(iteration_missing_values, iteration, class) = class_f1;
        end
    end
    hooi_coil_eval_object{iteration} = hooi_coil_eval_object_iteration;
    hooi_coil_eval_position{iteration} = hooi_coil_eval_position_iteration;
end

%% Display metrics - with position
% Global precision
coil_global_precisions
mean(coil_global_precisions')
%% Precision per class
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
display_tile = 11;
for m = 1:nb_iterations_for_missing_values
    nexttile
    precision_matrix = squeeze(coil_class_precisions(m, :, :));
    precision_matrix(isnan(precision_matrix)) = 0;
    heatmap([squeeze(precision_matrix); NaN(1, 10); mean(squeeze(precision_matrix))], 'Title', strcat('Class precision for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% Recall per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    recall_matrix = squeeze(coil_class_recalls(m, :, :));
    recall_matrix(isnan(recall_matrix)) = 0;
    heatmap([squeeze(recall_matrix); NaN(1, 10); mean(squeeze(recall_matrix))], 'Title', strcat('Class recall for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% F1-score per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    f1_matrix = squeeze(coil_class_f1s(m, :, :));
    f1_matrix(isnan(f1_matrix)) = 0;
    heatmap([squeeze(f1_matrix); NaN(1, 10); mean(squeeze(f1_matrix))], 'Title', strcat('Class F1-score for the coil dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end

%% Precision of position
coil_global_position_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 1);
degree_difference = 5;
position_difference = floor(degree_difference / 5);
for iteration = 1:nb_iterations
    for m = 1:nb_iterations_for_missing_values
        confusion_matrix = hooi_coil_eval_position{iteration};
        confusion_matrix = squeeze(confusion_matrix(m, :, :));
        correct_classification = 0;
        for i = 1:72
            start_index = i - 1;
            if start_index == 0
                start_index = 72;
            end
            end_index = i + 1;
            if end_index == 73
                end_index = 1;
            end
            for j = [start_index, i, end_index]
                correct_classification = correct_classification + confusion_matrix(i, j);
            end
        end
        coil_global_position_precisions(m, iteration) = correct_classification / sum(sum(confusion_matrix));
    end
end
coil_global_position_precisions
mean(coil_global_position_precisions')