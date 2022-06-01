%% Load images and construct tensor
% The first two dimensions of the tensor represent the pixels, and the
% third dimension represents the images.
att_data = zeros(112, 92, 400);
elt = 1;
for subject = 1:40
    for img = 0:9
        att_data(:, :, elt) = imread(strcat('datasets/att/',num2str(subject),'-00000',num2str(img),'.jpg'));
        elt = elt + 1;
    end
end
att_tensor = tensor(att_data);

%% Run HOOI and HALS-NTD decompositions
% The ranks of the decomposition have been chosen by keeping the number of 
% singular values that represent 95% of the sum of all singular values of 
% the matricized tensor on the concerned dimension
ranks = find_best_ranks(att_tensor, 0.95);
% HOOI decomposition
att_hooi = tucker_als2(att_tensor, ranks);
% HALS-NTD decomposition
opts = struct('lda_ortho', 1, 'init', 'eigs');
[~, att_fm, att_core, att_fit, ~] = tucker_localhals(att_tensor, ranks, opts);

%% Find clusters and compute accuracy
best_hooi_eval = zeros(40, 40);
best_hals_eval = zeros(40, 40);
best_hooi_accuracy = 0.0;
best_hals_accuracy = 0.0;
hooi_accuracies = zeros(100, 1);
hals_accuracies = zeros(100, 1);
for a = 1:100
    % Clusters for the HOOI 
    hooi_att_clusters = fkmeans(att_hooi.U{3}, 40);
    % Clusters for the HALS-NTD
    hals_att_clusters = fkmeans(att_fm{3}, 40);

    % Regroup the images of a same subject together
    hooi_att_eval = zeros(40, 40);
    hals_att_eval = zeros(40, 40);
    for i = 1:length(hooi_att_clusters)
        c = floor((i-1)/10);
        hooi_att_eval(c + 1, hooi_att_clusters(i)) = hooi_att_eval(c + 1, hooi_att_clusters(i)) + 1;
        hals_att_eval(c + 1, hals_att_clusters(i)) = hals_att_eval(c + 1, hals_att_clusters(i)) + 1;
    end
    % Put max value on diagonal
    for i = 1:40
        % For HOOI
        [val, idx] = sort(hooi_att_eval(i, :), 'descend');
        idx = idx(1);
        if (idx > i || (idx < i && val(1) > hooi_att_eval(idx, idx)))
            tmp = hooi_att_eval(:, i);
            hooi_att_eval(:, i) = hooi_att_eval(:, idx);
            hooi_att_eval(:, idx) = tmp;
        end
        % For HALS-NTD
        [val, idx] = sort(hals_att_eval(i, :), 'descend');
        idx = idx(1);
        if (idx > i || (idx < i && val(1) > hals_att_eval(idx, idx)))
            tmp = hals_att_eval(:, i);
            hals_att_eval(:, i) = hals_att_eval(:, idx);
            hals_att_eval(:, idx) = tmp;
        end
    end
    % Compute accuracy
    hooi_accuracy = sum(diag(hooi_att_eval)) / 400;
    hals_accuracy = sum(diag(hals_att_eval)) / 400;
    hooi_accuracies(a) = hooi_accuracy;
    hals_accuracies(a) = hals_accuracy;
    if (hooi_accuracy > best_hooi_accuracy)
        best_hooi_accuracy = hooi_accuracy;
        best_hooi_eval = hooi_att_eval;
    end
    if (hals_accuracy > best_hals_accuracy)
        best_hals_accuracy = hals_accuracy;
        best_hals_eval = hals_att_eval;
    end
end

%% Display results
heatmap(best_hooi_eval)
figure
heatmap(best_hals_eval)

disp('HOOI results: ')
disp(strcat("Minimum: ", num2str(min(hooi_accuracies))))
disp(strcat("Maximum: ", num2str(max(hooi_accuracies))))
disp(strcat("Mean: ", num2str(mean(hooi_accuracies))))
disp(strcat("Median: ", num2str(median(hooi_accuracies))))
disp(strcat("Standard deviation: ", num2str(std(hooi_accuracies))))
disp(strcat("Variance: ", num2str(var(hooi_accuracies))))

disp('HALS results: ')
disp(strcat("Minimum: ", num2str(min(hals_accuracies))))
disp(strcat("Maximum: ", num2str(max(hals_accuracies))))
disp(strcat("Mean: ", num2str(mean(hals_accuracies))))
disp(strcat("Median: ", num2str(median(hals_accuracies))))
disp(strcat("Standard deviation: ", num2str(std(hals_accuracies))))
disp(strcat("Variance: ", num2str(var(hals_accuracies))))
