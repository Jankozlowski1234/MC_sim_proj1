setwd("C:/Users/jaako/Desktop/studia/Monte Carlo/MC_sim_proj1")
library(ggplot2)

second_level_p_value<-function(p_val){
  n<-length(p_val)
  parts<-seq(0,1,0.1)
  R<-replicate(10,0.1*n)
  ilosc_w_przedzialach<-sapply(1:10, function(i){
    sum(p_val>=parts[i]&p_val<parts[i+1])
  })
  1-pchisq(sum((ilosc_w_przedzialach-R)^2/R),9)
}


dane<-read.csv('wyniki_zad1.csv',header = T)

dane$P.value<-as.numeric(dane$P.value)
dane$n<-as.numeric(dane$n)


ggplot(dane,aes(x = P.value,fill = Test,col = Test))+
  geom_histogram(position = "identity",alpha = 0.5)+facet_wrap(~Generator)+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "Histogtamy p_wartosci dla różnych testów",
       subtitle = "i różnych generatorów",
       x="P_wartość",y="")


p_values_second<-data.frame(matrix(ncol = 4, nrow = 4))

p_values_second[1,1]<-second_level_p_value(subset(dane,dane$Test=="KS"&
                                                    dane$Generator=="GLCG")$P.value)
p_values_second[2,1]<-second_level_p_value(subset(dane,dane$Test=="Chi"&
                                                    dane$Generator=="GLCG")$P.value)
p_values_second[3,1]<-second_level_p_value(subset(dane,dane$Test=="poker"&
                                                    dane$Generator=="GLCG")$P.value)
p_values_second[4,1]<-second_level_p_value(subset(dane,dane$Test=="Furier"&
                                                    dane$Generator=="GLCG")$P.value)

p_values_second[1,2]<-second_level_p_value(subset(dane,dane$Test=="KS"&
                                                    dane$Generator=="RC")$P.value)
p_values_second[2,2]<-second_level_p_value(subset(dane,dane$Test=="Chi"&
                                                    dane$Generator=="RC")$P.value)
p_values_second[3,2]<-second_level_p_value(subset(dane,dane$Test=="poker"&
                                                    dane$Generator=="RC")$P.value)
p_values_second[4,2]<-second_level_p_value(subset(dane,dane$Test=="Furier"&
                                                    dane$Generator=="RC")$P.value)

p_values_second[1,3]<-second_level_p_value(subset(dane,dane$Test=="KS"&
                                                    dane$Generator=="Marsagli")$P.value)
p_values_second[2,3]<-second_level_p_value(subset(dane,dane$Test=="Chi"&
                                                    dane$Generator=="Marsagli")$P.value)
p_values_second[3,3]<-second_level_p_value(subset(dane,dane$Test=="poker"&
                                                    dane$Generator=="Marsagli")$P.value)
p_values_second[4,3]<-second_level_p_value(subset(dane,dane$Test=="Furier"&
                                                    dane$Generator=="Marsagli")$P.value)

p_values_second[1,4]<-second_level_p_value(subset(dane,dane$Test=="KS"&
                                                    dane$Generator=="Ziff")$P.value)
p_values_second[2,4]<-second_level_p_value(subset(dane,dane$Test=="Chi"&
                                                    dane$Generator=="Ziff")$P.value)
p_values_second[3,4]<-second_level_p_value(subset(dane,dane$Test=="poker"&
                                                    dane$Generator=="Ziff")$P.value)
p_values_second[4,4]<-second_level_p_value(subset(dane,dane$Test=="Furier"&
                                                    dane$Generator=="Ziff")$P.value)


colnames(p_values_second)<-c("GLCG","RC(32)","Marsagli","Ziff")
rownames(p_values_second)<-c("KS","Chi","Poker","Fourier")



##zad2

narysuj_zad_2<-function(dane2){
  dane2$P.value <-as.numeric(dane2$P.value)
  dane2$n <-as.numeric(dane2$n)
  n <-dane2$n[1]
  
  p<-ggplot(dane2,aes(x   = P.value,fill = Liczba,col = Liczba))+
    geom_histogram(position = "identity",binwidth = 0.02)+facet_wrap(~Liczba)+
    theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))+ 
    labs(title = "Histogtamy p_wartosci dla różnych liczb",
         subtitle = paste("dla n =",n),
         x="P_wartość",y="")
  
  p_val<-data.frame(matrix(ncol = 3,nrow = 1))
  colnames(p_val)<-c("Pi","e","sqrt2")
  p_val[1]<-second_level_p_value(subset(dane2,dane2$Liczba =="Pi")$P.value)
  p_val[2]<-second_level_p_value(subset(dane2,dane2$Liczba =="e")$P.value)
  p_val[3]<-second_level_p_value(subset(dane2,dane2$Liczba =="sqrt2")$P.value)
  print(p)
  
  return(p_val)
}
dane2<-read.csv('wyniki_zad2_n_5000.csv',header = T)
r<-narysuj_zad_2(dane2)

