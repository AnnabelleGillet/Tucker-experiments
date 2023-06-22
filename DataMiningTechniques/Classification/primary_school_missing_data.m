%% Load file
fid = fopen('datasets/primaryschool.csv');
tline = fgetl(fid);
students = {};
times = {};
classes = {'1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B'};
time_granularity = 5;
while ischar(tline)
    line = split(tline, char(9));
    time = num2str(floor(str2num(line{1}) / (time_granularity * 60)));
    student1 = strcat(line{4}, {'-'}, line{2});
    student2 = strcat(line{5}, {'-'}, line{3});
    if ~any(strcmp(students, student1{1}))
        students{length(students)+1} = student1{1};
    end
    if ~any(strcmp(students, student2{1}))
        students{length(students)+1} = student2{1};
    end
    if ~any(strcmp(times, time))
        times{length(times)+1} = time;
    end
    tline = fgetl(fid);
end
fclose(fid);
students = sort(students);

%% Construct tensor data
% Two dimensions represent the persons, and the third dimension represent
% the time, with a granularity of 5 minutes. When a person p1 interacts
% with a person p2 at a time t, the value 1 is put as corresponding element
% indexed by p1, p2 and t. 
tensor_data = zeros(length(students), length(students), length(times));
fid = fopen('datasets/primaryschool.csv');
tline = fgetl(fid);
while ischar(tline)
    line = split(tline, char(9));
    time = num2str(floor(str2num(line{1}) / (time_granularity * 60)));
    student1 = strcat(line{4}, {'-'}, line{2});
    student2 = strcat(line{5}, {'-'}, line{3});
    index_student1 = find(strcmp(students, student1{1}));
    index_student2 = find(strcmp(students, student2{1}));
    index_time = find(strcmp(times, time));
    tensor_data(index_student1, index_student2, index_time) = 1.0;
    tensor_data(index_student2, index_student1, index_time) = 1.0;
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
nb_iterations_for_missing_values = 10;
ps_train_limit = 10;
hooi_ps_eval = {};
ps_global_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, 1);
ps_class_precisions = zeros(nb_iterations_for_missing_values, nb_iterations, length(classes));
ps_class_recalls = zeros(nb_iterations_for_missing_values, nb_iterations, length(classes));
ps_class_f1s = zeros(nb_iterations_for_missing_values, nb_iterations, length(classes));

for iteration = 1:nb_iterations
    fprintf('Iteration %2d\n', iteration);

    % Split data 
    ps_train_data = zeros((10 * primary_school_train_limit), length(students), length(times), length(classes));
    ps_test_data = zeros(length(students) - 10 - (10 * primary_school_train_limit), length(students), length(times), length(classes));
    for i = 1:length(classes)
        class = classes{i};
        students_of_class = find(~cellfun(@isempty,strfind(students, class)));
        start_index_per_iteration = (length(students_of_class) - ps_train_limit) / nb_iterations;
        start_index = floor(1 + ((iteration - 1) * start_index_per_iteration));
        ps_train_data((ps_train_limit * (i - 1)) + 1:ps_train_limit * i, :, :, i) = tensor_data(students_of_class(start_index:start_index + ps_train_limit - 1), :, :);
        if iteration > 1
            ps_test_data(1:start_index - 1, :, :, i) = tensor_data(students_of_class(1:start_index - 1), :, :);
        end
        if iteration < nb_iterations
            ps_test_data(start_index:length(students_of_class) - ps_train_limit, :, :, i) = tensor_data(students_of_class(start_index + ps_train_limit:length(students_of_class)), :, :);
        end
    end
    [max_students, ~, ~, ~] = size(ps_test_data);
    ps_train_tensor = tensor(ps_train_data);

    % Run HOOI decomposition
    ps_hooi = tucker_als2(ps_train_tensor, [10 10 4 10]);

    % Classification of test data
    hooi_ps_eval_iteration = zeros(nb_iterations_for_missing_values, 10, 10);
    initial_size = zeros(length(students) - 10 - (10 * primary_school_train_limit), 10);
    for iteration_missing_values = 1:nb_iterations_for_missing_values
        for class = 1:10
            fprintf(' Class %2d\n', class);
            for student = 1:max_students
                if sum(sum(ps_test_data(student, :, :, class))) > 0
                    data_to_classify = zeros((10 * ps_train_limit), length(students), length(times));
                    % Remove data
                    if iteration_missing_values == 1
                        initial_size(student, class) = length(find(tensor(ps_test_data(student, :, :, class))));
                    else
                        indexes = find(tensor(ps_test_data(student, :, :, class)));
                        indexes(:, 1) = [];
                        indexes = [ones(length(indexes), 1)*student indexes ones(length(indexes), 1)*class];
                        number_of_elements_to_remove = floor(0.1 * initial_size(student, class));
                        selected_indexes = indexes(randperm(length(indexes), number_of_elements_to_remove), :);
                        tmp = tensor(ps_test_data);
                        tmp(selected_indexes) = 0.0;
                        ps_test_data = tmp.data;
                    end
                    %
                    data_to_classify(:, :, :) = repmat(ps_test_data(student, :, :, class), (10 * primary_school_train_limit), 1);
                    data_to_classify = tensor(data_to_classify, [(10 * ps_train_limit), length(students), length(times), 1]);
                    data_to_classify = ttm(data_to_classify, ps_hooi.U{1}', 1);
                    data_to_classify = ttm(data_to_classify, ps_hooi.U{2}', 2);
                    data_to_classify = ttm(data_to_classify, ps_hooi.U{3}', 3);
                    min_n = realmax("double");
                    best_class = 0;
                    for class_to_try = 1:10
                        data_to_try = ttm(ps_hooi.core, ps_hooi.U{4}(class_to_try, :), 4);
                        n = norm(data_to_try - data_to_classify);
                        if n < min_n
                            best_class = class_to_try;
                            min_n = n;
                        end
                    end
                    hooi_ps_eval_iteration(iteration_missing_values, class, best_class) = hooi_ps_eval_iteration(iteration_missing_values, class, best_class) + 1;
                end
            end
        end
    % Computation of metrics
    ps_global_precisions(iteration_missing_values, iteration) = trace(squeeze(hooi_ps_eval_iteration(iteration_missing_values, :, :)))/sum(sum(squeeze(hooi_ps_eval_iteration(iteration_missing_values, :, :))));
    for class = 1:10
        class_precision = hooi_ps_eval_iteration(iteration_missing_values, class, class) / sum(hooi_ps_eval_iteration(iteration_missing_values, :, class));
        class_recall = hooi_ps_eval_iteration(iteration_missing_values, class, class) / sum(hooi_ps_eval_iteration(iteration_missing_values, class, :));
        class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall);
        ps_class_precisions(iteration_missing_values, iteration, class) = class_precision;
        ps_class_recalls(iteration_missing_values, iteration, class) = class_recall;
        ps_class_f1s(iteration_missing_values, iteration, class) = class_f1;
    end
    end
    hooi_ps_eval{iteration} = hooi_ps_eval_iteration;
end

%% Display metrics
% Global precision
ps_global_precisions
mean(ps_global_precisions')
%% Precision per class
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
display_tile = 11;
for m = 1:nb_iterations_for_missing_values
    nexttile
    precision_matrix = squeeze(ps_class_precisions(m, :, :));
    precision_matrix(isnan(precision_matrix)) = 0;
    heatmap([squeeze(precision_matrix); NaN(1, 10); mean(squeeze(precision_matrix))], 'Title', strcat('Class precision for the primary school dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% Recall per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    recall_matrix = squeeze(ps_class_recalls(m, :, :));
    recall_matrix(isnan(recall_matrix)) = 0;
    heatmap([squeeze(recall_matrix); NaN(1, 10); mean(squeeze(recall_matrix))], 'Title', strcat('Class recall for the primary school dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
%% F1-score per class
figure
t = tiledlayout(ceil(nb_iterations_for_missing_values / 2), 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations_for_missing_values
    nexttile
    f1_matrix = squeeze(ps_class_f1s(m, :, :));
    f1_matrix(isnan(f1_matrix)) = 0;
    heatmap([squeeze(f1_matrix); NaN(1, 10); mean(squeeze(f1_matrix))], 'Title', strcat('Class F1-score for the primary school dataset, ', num2str((m - 1) * 10), '% of missing values'), ...
        'YLabel', 'Iteration', 'ColorbarVisible', 'off', 'YData', {'1', '2', '3', '4', '5', '.', 'Mean'})
end
