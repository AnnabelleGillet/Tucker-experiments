%% Load file
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
primary_school_tensor = tensor(tensor_data);

%%%%%%%%%% HOOI %%%%%%%%%%
%% Run HOOI decomposition
primary_school_hooi = tucker_als2(primary_school_tensor, [13, 13, 4]);

%% Visualize factor matrices
% The factor matrices representing the persons are visualized as heatmap,
% in which the students are sorted according to their class (first the
% stdents of class 1A, then 1B, and so on until 5B). The teachers are put
% at the end. 
figure
plot_time(primary_school_hooi.U{3});
figure
[~, idxs] = sort(students);
heatmap((primary_school_hooi.U{1}(idxs, :) ./ max(abs(primary_school_hooi.U{1})))', 'ColorMap', jet, 'xData', sort(students));
figure
heatmap((primary_school_hooi.U{2}(idxs, :) ./ max(abs(primary_school_hooi.U{2})))', 'ColorMap', jet, 'xData', sort(students));

%% Visualize best ranks
% The best ranks are the vectors of each factor matrices that index the
% highest values of the core tensor.
sorted_primary_school_core_tensor = sort_core_tensor(primary_school_hooi.core.data);
for i = 1:5
    figure
    see(primary_school_hooi, sorted_primary_school_core_tensor{i, 2}, students);
end

%% Visualize best ranks for a specific rank on dimension 1
% Show the best ranks when we consider only the sub core tensor that
% corresponds to rank selected on the first dimension.
rank_to_see = 9;
sorted_primary_school_core_tensor = sort_core_tensor(primary_school_hooi.core.data(rank_to_see, :, :));
total = sum(abs(cell2mat(sorted_primary_school_core_tensor(:, 1))));
somme = 0.0;
i = 1;
while (somme < total * 0.5)
    figure
    somme = somme + abs(sorted_primary_school_core_tensor{i, 1});
    ranks_to_see = sorted_primary_school_core_tensor{i, 2};
    ranks_to_see(1) = rank_to_see;
    see(primary_school_hooi, ranks_to_see, students);
    i = i + 1;
end

%%%%%%%%%% HALS %%%%%%%%%%
%% Run HALS decomposition
opts = struct('lda_ortho', 1, 'init', 'eigs');
[~, ps_fm, ps_core, ps_fit, ~] = tucker_localhals(primary_school_tensor, [13, 13, 4], opts);

%% Visualize factor matrices
% The factor matrices representing the persons are visualized as heatmap,
% in which the students are sorted according to their class (first the
% stdents of class 1A, then 1B, and so on until 5B). The teachers are put
% at the end. 
figure
plot_time(ps_fm{3});
figure
[~, idxs] = sort(students);
heatmap((ps_fm{1}(idxs, :) ./ max(ps_fm{1}))', 'ColorMap', jet, 'xData', sort(students));
figure
heatmap((ps_fm{2}(idxs, :) ./ max(ps_fm{2}))', 'ColorMap', jet, 'xData', sort(students));

%% Visualize best ranks
% The best ranks are the vectors of each factor matrices that index the
% highest values of the core tensor.
sorted_primary_school_core_tensor = sort_core_tensor(ps_core.data);
for i = 1:5
    figure
    see(ttensor(ps_core, ps_fm), sorted_primary_school_core_tensor{i, 2}, students);
end

%% Visualize best ranks for a specific rank on dimension 1
% Show the best ranks when we consider only the sub core tensor that
% corresponds to rank selected on the first dimension.
rank_to_see = 8;
sorted_primary_school_core_tensor = sort_core_tensor(ps_core.data(rank_to_see, :, :));
total = sum(cell2mat(sorted_primary_school_core_tensor(:, 1)));
somme = 0.0;
i = 1;
while (somme < total * 0.99)
    figure
    somme = somme + sorted_primary_school_core_tensor{i, 1};
    ranks_to_see = sorted_primary_school_core_tensor{i, 2};
    ranks_to_see(1) = rank_to_see;
    see(ttensor(ps_core, ps_fm), ranks_to_see, students);
    i = i + 1;
end
