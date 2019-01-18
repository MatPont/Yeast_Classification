setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S1/Data_Science/BD_projet_2018_2019")

dfYeast <- read.table("yeast.data")
dfYeast
names(dfYeast) <- c("Sequence Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "Localization (label)")
dfYeast
summary(dfYeast)
#matYeast <- data.matrix(dfYeast)
#matYeast <- matYeast[,-1]
matYeast <- dfYeast[,-1]
matYeast
varYeast <- matYeast[,-9]
varYeast
for(i in 1:ncol(varYeast)){
  print(sd(varYeast[,i]))
}
mcor <- cor(varYeast)
mcor
write.csv(mcor,"yeast_correlation-matrix.csv")
boxplot(varYeast)

# ALL PCA
library("FactoMineR")
library("corrplot")
library("factoextra")
help(PCA)

matYeast

valeurs_abberantes <- function(x){
  tmp <- ncol(x) -1 
  for(i in 1:tmp){
    vector_indice=c()
    q1=quantile(x[,i])[2]
    q3=quantile(x[,i])[4]
    ecart = q3-q1
    val_extremes_min = q1-1.5*ecart
    val_extremes_max = q3+1.5*ecart
    for (j in 1:nrow(x)){
      if (x[j,i]>val_extremes_max || x[j,i]<val_extremes_min){
        vector_indice = c(vector_indice,j)
      }
    }
    vector_indice=unique(vector_indice)
    print(length(vector_indice))
    x = x[-c(vector_indice),]
    print(x)
  }
  return(x)
}
matYeast <- valeurs_abberantes(matYeast)
matYeast

# PCA 
layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(matYeast, quali.sup = c(5, 9), scale.unit = FALSE)
plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")

var <- get_pca_var(resPCA)
corrplot(var$cos2)
corrplot(var$contrib, is.corr=FALSE)

# PCA avec pox en variable supplémentaire
layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(matYeast, quali.sup = c(5, 9), quanti.sup = c(6), scale.unit = FALSE)
plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")

# PCA avec pox, vac, nuc et alm en variables supplémentaires
layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(matYeast, quali.sup = c(5, 9), quanti.sup = c(3,6,7,8), scale.unit = FALSE)

resPCA$eig
dimdesc(resPCA)

plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")
fviz_pca_ind(resPCA, col.ind = matYeast[,9], label = "none", addEllipses = TRUE)

