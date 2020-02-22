Array=csvread("../dataset/housing_dataset_matlab.csv");

fields = ["SalePrice","LotFrontage",  "LotArea", "OverallQual", "MasVnrArea", "YearBuilt", "BsmtUnfSF", "YearRemodAdd", "TotalBsmtSF", "BsmtFinSF1", "1stFlrSF"];
col1 = Array(:,1);

for i = 2:11
    col2 = Array(:,i);

    scatter(col2, col1);
    title(strcat(fields{1}," Vs. ",fields{i}));
    xlabel(fields{i})
    ylabel(fields{1})
    saveas(gcf, strcat("../figures/",fields{i},".jpg"));
    
end
