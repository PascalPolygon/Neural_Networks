Array=csvread("../dataset/housing_dataset_matlab.csv");
col1 = Array(:,1);
for i = 1:11
    col2 = Array(:,i);
    saveas(scatter(col1, col2), strcat("scatter_",int2str(i),".png"));
end
