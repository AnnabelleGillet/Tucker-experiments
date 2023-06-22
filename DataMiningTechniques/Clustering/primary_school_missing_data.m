%% Load file
% Read the file a first time to know how many elements we have on each
% dimension.
fid = fopen('datasets/primaryschool.csv');
tline = fgetl(fid);
students = {};
times = {};
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

%% Construct tensor
% Two dimensions reprsent the persons, and the third dimension represent
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
ps_tensor = tensor(tensor_data);

% Keep class of students
students_real_class = zeros(242, 1);
for i = 1:5
    students_of_class = find(contains(students, strcat(num2str(i), 'A')));
    students_real_class(students_of_class) = (i - 1) * 2 + 1;
    students_of_class = find(contains(students, strcat(num2str(i), 'B')));
    students_real_class(students_of_class) = i * 2;
end
students_of_class = find(contains(students, 'Teacher'));
students_real_class(students_of_class) = 11;

%% Evaluate Tucker clustering with missing data for some students
% Select some students for each classes and randomly remove data only for
% these students, to see if they are still clustered with other students of
% their class.
ps_tensor_missing_student_data = ps_tensor;
number_of_selected_students = 5;
chosen_student_indexes = zeros(10, number_of_selected_students);
initial_size = zeros(10, number_of_selected_students);
for i = 1:5
    students_of_class = find(contains(students, strcat(num2str(i), 'A')));
    chosen_student_indexes((i - 1) * 2 + 1, :) = students_of_class(1:number_of_selected_students);
    for j = 1:number_of_selected_students
        initial_size((i - 1) * 2 + 1, j) = length(find(ps_tensor_missing_student_data.data(students_of_class(j), :, :)));
    end
    students_of_class = find(contains(students, strcat(num2str(i), 'B')));
    chosen_student_indexes(i * 2, :) = students_of_class(1:number_of_selected_students);
    for j = 1:number_of_selected_students
        initial_size(i * 2, j) = length(find(ps_tensor_missing_student_data.data(students_of_class(j), :, :)));
    end
end
nb_iterations = 10;
ps_missing_student_data_eval = zeros(nb_iterations, 10, 10);
ps_missing_chosen_student_data_eval = zeros(nb_iterations, 10, 10);
% 1 - Global, 2 - Only for chosen students
ps_missing_student_data_precision = zeros(nb_iterations, 2);
ps_missing_student_data_ri = zeros(nb_iterations, 2);
ps_missing_student_data_ari = zeros(nb_iterations, 2);

for m = 1:nb_iterations
    fprintf('Iteration %2d\n', m);
    % Randomly remove data for each selected student
    if m > 1
        for i = 1:10
            for j = 1:number_of_selected_students
                indexes = find(ps_tensor_missing_student_data(chosen_student_indexes(i, j), :, :));
                indexes = [ones(length(indexes), 1)*chosen_student_indexes(i, j) indexes];
                number_of_elements_to_remove = round(0.1 * initial_size(i, j));
                selected_indexes = indexes(randperm(length(indexes), number_of_elements_to_remove), :);
                ps_tensor_missing_student_data(selected_indexes) = 0.0;
            end
        end
    end

    % Run HOOI decomposition
    ps_missing_student_data_hooi = tucker_als2(ps_tensor_missing_student_data, [13 13 4]);
    
    % Find clusters with k-medoids
    hooi_ps_missing_student_data_clusters = Kmedoids(ps_missing_student_data_hooi.U{1}(1:232, :), 10);
    
    % Build confusion matrix
    for i = 1:length(hooi_ps_missing_student_data_clusters)
        class = students_real_class(i);
        ps_missing_student_data_eval(m, class, hooi_ps_missing_student_data_clusters(i)) = ps_missing_student_data_eval(m, class, hooi_ps_missing_student_data_clusters(i)) + 1;
        if any(chosen_student_indexes(:) == i)
            ps_missing_chosen_student_data_eval(m, class, hooi_ps_missing_student_data_clusters(i)) = ps_missing_chosen_student_data_eval(m, class, hooi_ps_missing_student_data_clusters(i)) + 1;
        end
    end
    
    % Put max value on diagonal
    for i = 1:10
        [val, idx] = sort(ps_missing_student_data_eval(m, i, :), 'descend');
        idx = idx(1);
        if (idx > i || (idx < i && val(1) > ps_missing_student_data_eval(m, idx, idx)))
            tmp = ps_missing_student_data_eval(m, :, i);
            ps_missing_student_data_eval(m, :, i) = ps_missing_student_data_eval(m, :, idx);
            ps_missing_student_data_eval(m, :, idx) = tmp;

            tmp = ps_missing_chosen_student_data_eval(m, :, i);
            ps_missing_chosen_student_data_eval(m, :, i) = ps_missing_chosen_student_data_eval(m, :, idx);
            ps_missing_chosen_student_data_eval(m, :, idx) = tmp;
        end
    end
    
    % Precision
    ps_missing_student_data_precision(m, 1) = sum(diag(squeeze(ps_missing_student_data_eval(m, :, :)))) / sum(sum(ps_missing_student_data_eval(m, :, :)));
    ps_missing_student_data_precision(m, 2) = sum(diag(squeeze(ps_missing_chosen_student_data_eval(m, :, :)))) / sum(sum(ps_missing_chosen_student_data_eval(m, :, :)));
    
    % Rand Index and Adjusted Rand Index
    ps_missing_student_data_ri(m, 1) = rand_index(hooi_ps_missing_student_data_clusters, students_real_class(1:232, 1));
    ps_missing_student_data_ri(m, 2) = rand_index(hooi_ps_missing_student_data_clusters(chosen_student_indexes(:)), students_real_class(chosen_student_indexes(:), 1));
    ps_missing_student_data_ari(m, 1) = adjusted_rand_index(squeeze(ps_missing_student_data_eval(m, :, :)));
    ps_missing_student_data_ari(m, 2) = adjusted_rand_index(squeeze(ps_missing_chosen_student_data_eval(m, :, :)));
end

%% Display result
t = tiledlayout(nb_iterations, 2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for m = 1:nb_iterations
    nexttile
    heatmap(squeeze(ps_missing_student_data_eval(m, :, :)), 'XLabel', 'Cluster found', 'YLabel', 'Real class', 'Title', strcat(num2str((m - 1) * 10), '% of missing data'))
    colorbar off
    nexttile
    heatmap(squeeze(ps_missing_chosen_student_data_eval(m, :, :)), 'XLabel', 'Cluster found', 'YLabel', 'Real class', 'Title', strcat(num2str((m - 1) * 10), '% of missing data, selected students only'))
    colorbar off
end
ps_missing_student_data_precision
ps_missing_student_data_ri
ps_missing_student_data_ari

