function [ dirList ] = get_directory_names( dir_name )

    %get_directory_names; this function outputs a cell with directory names (as
    %strings), given a certain dir name (string)
    dd = dir(dir_name);
    isub = [dd(:).isdir]; %# returns logical vector
    dirList = {dd(isub).name}';
    dirList(ismember(dirList,{'.','..'})) = [];

end