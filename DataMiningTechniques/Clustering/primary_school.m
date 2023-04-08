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

%% Find best ranks
find_best_ranks(ps_tensor, 0.8);

%% Run HOOI decomposition
ps_hooi = tucker_als2(ps_tensor, [13 13 4]);

%% Find clusters with k-medoids
hooi_ps_clusters = Kmedoids(ps_hooi.U{1}(1:232, :), 10);

%% Build confusion matrix
students_real_class = zeros(242, 1);
for i = 1:5
    students_of_class = find(contains(students, strcat(num2str(i), 'A')));
    students_real_class(students_of_class) = (i - 1) * 2 + 1;
    students_of_class = find(contains(students, strcat(num2str(i), 'B')));
    students_real_class(students_of_class) = i * 2;
end
students_of_class = find(contains(students, 'Teacher'));
students_real_class(students_of_class) = 11;

ps_eval = zeros(10, 10);
for i = 1:length(hooi_ps_clusters)
    class = students_real_class(i);
    ps_eval(class, hooi_ps_clusters(i)) = ps_eval(class, hooi_ps_clusters(i)) + 1;
end

% Put max value on diagonal
for i = 1:10
    [val, idx] = sort(ps_eval(i, :), 'descend');
    idx = idx(1);
    if (idx > i || (idx < i && val(1) > ps_eval(idx, idx)))
        tmp = ps_eval(:, i);
        ps_eval(:, i) = ps_eval(:, idx);
        ps_eval(:, idx) = tmp;
    end
end

%% Show result
hooi_precision = sum(diag(ps_eval)) / length(hooi_ps_clusters)
heatmap(ps_eval, 'XLabel', 'Cluster found', 'YLabel', 'Real class')
