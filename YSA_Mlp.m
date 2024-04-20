clc
clear all
close all
df = readtable("heart.csv");


age = df.Age;
cinsiyet = df.Sex;
%cinsiyet verisindeki M ve F olan cinsiyet verilerini 0 1 ile değiştirmek
cinsiyet_values = df.Sex;
cinsiyet_values(strcmp(cinsiyet_values, 'M')) = {'0'};
cinsiyet_values(strcmp(cinsiyet_values, 'F')) = {'1'};
cinsiyet_values = str2double(cinsiyet_values);
%gogus agrısı tiplerini 0,1,2,3 olarak sınıflandırıp veritipini string
%olmasın diye double cevirme
gogusagri=df.ChestPainType;
gogusagri_values=df.ChestPainType;
gogusagri_values(strcmp(gogusagri_values,'NAP'))={'0'};
gogusagri_values(strcmp(gogusagri_values,'ASY'))={'1'};
gogusagri_values(strcmp(gogusagri_values,'ATA'))={'2'};
gogusagri_values(strcmp(gogusagri_values,'TA'))={'3'};
gogusagri_values = str2double(gogusagri_values);
%ekg verilerini sayısal veri olarak sınıflandırma   
ekg=df.RestingECG;
ekg_values=df.RestingECG;
ekg_values(strcmp(ekg_values,'Normal'))={'0'};
ekg_values(strcmp(ekg_values,'LVH'))={'1'};
ekg_values(strcmp(ekg_values,'ST'))={'2'};
ekg_values= str2double(ekg_values);
%egzersiz yapma durumunu sınıflandırma
egzersiz=df.ExerciseAngina;
egzersiz_values=df.ExerciseAngina;
egzersiz_values(strcmp(egzersiz_values,'Y'))={'1'};
egzersiz_values(strcmp(egzersiz_values,'N'))={'0'};
egzersiz_values=str2double(egzersiz_values);
%st_slope verisini numerik veri yapma
st=df.ST_Slope;
st_values=df.ST_Slope;
st_values(strcmp(st_values,'Up'))={'0'};
st_values(strcmp(st_values,'Flat'))={'1'};
st_values=str2double(st_values);


%verileri kategorik veriden sayılsal veriye cevirdikten sonra halihazırda
%sayısal olan verileri de doubleye cevirdikten sonra tekrardan tablo
%olusturma işlemi (çıktı verileri haric)
X = [double(age), cinsiyet_values,gogusagri_values,double(df.RestingBP),double(df.Cholesterol),double(df.FastingBS),ekg_values, double(df.MaxHR),egzersiz_values,double(df.Oldpeak),st_values];
%çıktı verilerini değişkene atama
y = df.HeartDisease;
%girdi verileri normalize etme(etkin öğrenme icin)
X = normalize(X);

%verileri pythondaki random state fonksiyonu işlevini gören algoritma ile
%rastgele parçalama işlemi(öğrenme kısmında birden fazla kodu calıştırma işlemi yapılmalı verilerin 1 classta toplanma ihtimaline karşı) 
rng(42); 
indices = randperm(size(X, 1));
split_point = round(0.5 * size(X, 1));
X_train = X(indices(1:split_point), :);
X_test = X(indices(split_point+1:end), :);
y_train = y(indices(1:split_point));
y_test = y(indices(split_point+1:end));

%ogrenilecek olan verilerin min ve max değerlerini bulma 
m1=min(X_train)';
m2=max(X_train)';
%max ve min değerlere gore sınırları belirleme
range = [m1 m2]; 
%12 22 1 olmak üzere 3 katmanlı model olusturma(logsig kullanma nedenimiz
%cıktı verilerimiz 0 veya 1 olması 2 cıktıdan fazla olacak olsaydı tansig
%kullanmak daha dogru sonucları verirdi)
net = newff(range,[11 22 1],{ 'tansig','tansig','logsig' }, 'trainlm' );
net.trainparam.show = 25; 
net.trainparam.epochs = 1000;
net.trainparam.goal = 1e-12;
net.trainparam.maxfail = 60;
net.trainparam.memreduc = 1;
net = train(net,X_train',y_train'); 
% view(net); 
y = net(X_train'); 
perf = perform(net,y,y_train)


plotconfusion(y_train',y)
title("train Data")

%% test data
res_test = net(X_test'); 
perf = perform(net,res_test,y_test)

figure

plotconfusion(y_test',res_test)
title("test Data")






