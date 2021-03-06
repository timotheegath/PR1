close all
clear all
load('distortion_example_image_checkpoints.mat')
how_many_faces = size(example_image, 1);
figure()
intervals = checkpoints;
for j = 1:how_many_faces
    for i = 1:length(intervals)
        subplot(how_many_faces, length(intervals), (j-1)*length(intervals) +i)
        image = reshape(example_image(j, :, i), [56, 46]);
        imagesc(image) 
        colormap(gray(256))
        set(gca,'XTick',[], 'YTick', [])
        daspect([1 1 1])
        title(['Face ', num2str(j), ' M=',num2str(intervals(i))])
    end
    
end
    