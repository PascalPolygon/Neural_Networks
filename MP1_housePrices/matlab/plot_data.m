Array=csvread("../dataset/housing_dataset_matlab.csv");

fields = ["SalePrice", "LotFrontage", "OverallQual", "MasVnrArea", "YearBuilt", "BsmtUnfSF", "YearRemodAdd", "TotalBsmtSF", "BsmtFinSF1", "1stFlrSF"];
col1 = Array(:,1);

for i = 1:10
    col2 = Array(:,i);

    scatter(col1, col2);
    title(strcat(fields{i}," Vs. ",fields{1}));
    xlabel(fields{i})
    ylabel(fields{1})
    saveas(gcf, strcat("../figures/",fields{i},".jpg"));
    
end
