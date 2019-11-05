clear all;
close all;
clc;
file_list = getAllFiles('Train');
[P, t]  = size(file_list);
[row, column] = size(imread(file_list{1}));
inp_mat = zeros(row*column, P);

lable = file_list;
for i = 1:P
    img = imread(file_list{i});
    inp_mat(:, i) = img(:);
end

mean_vector = zeros(row*column,1); 
for i = 1:row*column
    mean_vector(i) = mean(inp_mat(i,:));
end
% createing mean face subtracted delta matrix
delta = zeros(row*column, P);
for i = 1:P
    delta(:,i) = inp_mat(:,i) - mean_vector;
end
size(delta);
% covariavnce matrix calculation
size(mean_vector);
cov_mat = zeros(P, P);

cov_mat = cov(delta);
[V, D] = eig(cov_mat);
% select k numnber of lesser significant directions
k = 15;
V = V(:, k+1:P);
bnm = size(V);
D = D(k+1:P, :);
feature_vector = V;
size(feature_vector);
eigen_face = feature_vector'*delta';
eigen_face_size = size(eigen_face);

signature = eigen_face*delta;
sig_size = size(signature);

% testing

file_list_test = getAllFiles('Test');
[T, t] = size(file_list_test);
success = 0;

for i = 1:T
    img = imread(file_list_test{i});
    img = img(:);
    img = double(img) - mean_vector;
    test_signature = eigen_face*img;
    distance = zeros(P, 1);
    for j = 1:P
        distance(j) = sum((test_signature - signature(:, j)).^2);
    end
    size(distance);
    [k, ind] = min(distance); 
    test_lable = strsplit(file_list_test{i}, '\');
    true_lable = strsplit(file_list{ind}, '\');
    te = test_lable{2};
    tr = true_lable{2};
    te(1:end-5);
    tr(1:end-5);
   % calculating accuracy   
   if (strcmp(te(1:end-5), tr(1:end-5)))
       success = success + 1;
   end
end
accuracy = (success / T)*100