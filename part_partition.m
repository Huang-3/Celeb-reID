
name = 'gallery';
path = ['./',name,'/'];

new_path1 = ['./' name, '_1_1/'];
if ~exist(new_path1)
    mkdir(new_path1);
end

new_path2 = ['./' name, '_1_2/'];
if ~exist(new_path2)
    mkdir(new_path2);
end

new_path3 = ['./' name, '_1_3/'];
if ~exist(new_path3)
    mkdir(new_path3);
end

new_path4 = ['./' name, '_2_1/'];
if ~exist(new_path4)
    mkdir(new_path4);
end

new_path5 = ['./' name, '_2_2/'];
if ~exist(new_path5)
    mkdir(new_path5);
end

im_list = dir([path, '*.jpg']);

for i = 1:length(im_list)
    i
    im = imread([path, im_list(i).name]);
    
    [h, w, c] = size(im);
    
    im_1_1 = im(1:ceil(h/3), :, :);
    im_1_2 = im(ceil(h/3):ceil(2*h/3), :, :);
    im_1_3 = im(ceil(2*h/3):end, :, :);
    
    im_2_1 = im(1:ceil(h/2), :, :);
    im_2_2 = im(ceil(h/2):end, :, :);
    

    imwrite(imresize(im_1_1, [224, 224]), [new_path1, im_list(i).name]);
    imwrite(imresize(im_1_2, [224, 224]), [new_path2, im_list(i).name]);
    imwrite(imresize(im_1_3, [224, 224]), [new_path3, im_list(i).name]);
    imwrite(imresize(im_2_1, [224, 224]), [new_path4, im_list(i).name]);
    imwrite(imresize(im_2_2, [224, 224]), [new_path5, im_list(i).name]);
    
end