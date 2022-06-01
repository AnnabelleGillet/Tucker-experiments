function [] = see(X, indices, students)
% Provide visualization for the primary school dataset 
% from the given indices on each dimension
vector_to_plot = X.U{3}(:,indices(3));
granularity = 5;
day1 = vector_to_plot(1:length(vector_to_plot)/2,:);
day2 = vector_to_plot(1+(length(vector_to_plot)/2):end,:);

subplot(2, 2, 1);
t = datetime(0,0,0,8,30,0):minutes(granularity):datetime(0,0,0,17,5,0);
plot(t,day1);
datetick('x','HH:MM');
title('Day 1');

subplot(2, 2, 2);
plot(t,day2);
datetick('x','HH:MM');
title('Day 2');

plotable_students1 = matrix_factor_to_plot(X.U{1}, students);
plotable_students2 = matrix_factor_to_plot(X.U{2}, students);

subplot(2, 2, [3, 4]);
heatmap([plotable_students1(:,indices(1)), plotable_students2(:,indices(2))]', 'xData', sort(students), 'yData', ['Student1'; 'Student2'], 'Colormap', jet);
end