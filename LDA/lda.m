clear all;
close all;
clc;
class_list = get_directory_names('Train');
[num_class, t] = size(class_list);
% assuming that each class has equal number of sample images
P = 1; % P is the total number of images across all classes
inp_mat = [];
label = [];
for i = 1:num_class
    file_list = getAllFiles(strcat('Train\', class_list{i}));
    [num_img, t] = size(file_list);
    file_list;
    label = [label file_list];
    for j = 1:num_img
        img = imread(file_list{j});
        inp_mat = [inp_mat img(:)];
    end
end
size(inp_mat);

mean_vector = mean(inp_mat, 2); 

% createing mean face subtracted delta matrix
[rows, P] =  size(inp_mat);
delta = zeros(rows, P);
for i = 1:P
    delta(:,i) = double(inp_mat(:,i)) - mean_vector;
end
size(delta);

% covariavnce matrix calculation
cov_mat = cov(delta);

[V, D] = eig(cov_mat);
% select k numnber of lesser significant directions
k = 15;
V = V(:, k+1:P);
D = D(k+1:P, :);
feature_vector = V;
size(feature_vector);
eigen_face = feature_vector'*delta';
size(eigen_face);

signature = eigen_face*delta;
sig_size = size(signature);

% now we apply LDA on the data
% we compute class means

class_mean_vector = [];
Sw = zeros(P-k, P-k);

for i = 1:num_class
    class_mean_vector =[class_mean_vector mean(signature(:, i:(i + num_img - 1)), 2)];
    Sw = Sw + cov(double(signature(:, i:(i + num_img - 1))'));
end
size(class_mean_vector);
MEAN = mean(class_mean_vector, 2);
size(MEAN);

Sb = zeros(P - k, P - k ); 
for i = 1:num_class 
     Sb = Sb + num_img.*(class_mean_vector(:, i) - MEAN)*(class_mean_vector(:, i) - MEAN)';
end
size(Sw);
size(Sb);

J = inv(Sw)*Sb;

[V, D] = eig(J);
% select m numnber of lesser significant directions
size(V);
size(D);
m = 1;
V = V(:, m+1:(P-k));
D = D(m+1:(P-k), :);
size(V);
size(D);

ficher_faces = V'*signature;
size(ficher_faces);

% testing

file_list_test = getAllFiles('Test');
[T, t] = size(file_list_test);
success = 0;

for i = 1:T
    img = imread(file_list_test{i});
    img = img(:);
    img = double(img) - mean_vector;
    PEF = eigen_face*img;
    size(PEF);
    proj_ficher_test_img = V'*PEF;
    size(proj_ficher_test_img);
    
    distance = zeros(P, 1);
    for j = 1:P
        distance(j) = sum((proj_ficher_test_img - ficher_faces(:, j)).^2);
    end
    size(distance);
    [k, ind] = min(distance); 
    test_lable = strsplit(file_list_test{i}, '\');
    true_lable = strsplit(label{ind}, '\');
    te = test_lable{2};
    tr = true_lable{2};
    te(1:end-5);
  
   % calculating accuracy   
   if (strcmp(te(1:end-5), tr))
       success = success + 1;
   end
end

accuracy = (success / T)*100